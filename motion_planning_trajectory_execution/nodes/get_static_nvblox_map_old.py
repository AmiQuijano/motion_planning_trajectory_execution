import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import torch
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

# cuRobo
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

# NVBlox
from curobo.wrap.model.robot_world import RobotWorld
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.state import TensorDeviceType
from curobo.geom.types import Cuboid
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

class RealsenseToBaseLinkTSDF(Node):
    def __init__(self, voxel_size=0.02, clipping_distance=1.0, device='cuda:0'):
        super().__init__('realsense_to_baselink_tsdf')
        self.voxel_size = voxel_size
        self.clipping_distance = clipping_distance
        self.device = device
        self.tensor_args = TensorDeviceType(device=self.device)

        # ROS2 TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Realsense interface
        self.realsense = RealsenseDataloader(clipping_distance_m=self.clipping_distance)

        # Create CuRobo world_model
        # For simplicity, we create a blank world config (no obstacles)
        world_cfg_dict = {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "tsdf",
                    "voxel_size": self.voxel_size,
                }
            }
        }
        self.world_model = RobotWorld(
            world_config=world_cfg_dict,
            collision_checker_type=CollisionCheckerType.BLOX,
            tensor_device_type=self.tensor_args
        )

    def get_camera_pose_base_link(self, frame_id='camera_depth_optical_frame', target_frame='camera_link'):
        """
        Look up the current camera pose in the base_link frame.
        Returns: curobo.types.math.Pose
        """
        try:
            tf_stamped: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame, frame_id, rclpy.time.Time()
            )
            t = torch.tensor([
                tf_stamped.transform.translation.x,
                tf_stamped.transform.translation.y,
                tf_stamped.transform.translation.z
            ], device=self.device)
            q = torch.tensor([
                tf_stamped.transform.rotation.w,
                tf_stamped.transform.rotation.x,
                tf_stamped.transform.rotation.y,
                tf_stamped.transform.rotation.z
            ], device=self.device)
            return Pose(position=t, quaternion=q)
        except:
            return None

    def clip_camera_edges(self, camera_data, h_ratio=0.05, w_ratio=0.05):
        """Clip depth edges to remove sensor noise."""
        depth_tensor = camera_data["depth"]
        h, w = depth_tensor.shape
        depth_tensor[: int(h_ratio * h), :] = 0.0
        depth_tensor[int((1 - h_ratio) * h):, :] = 0.0
        depth_tensor[:, : int(w_ratio * w)] = 0.0
        depth_tensor[:, int((1 - w_ratio) * w):] = 0.0

    def update_tsdf_from_camera(self):
        """Grab a frame and integrate it into the TSDF."""
        data = self.realsense.get_data()
        self.clip_camera_edges(data)
        camera_pose = self.get_camera_pose_base_link()
        if camera_pose is None:
            return

        data_camera = CameraObservation(
            depth_image=data["depth"],
            intrinsics=data["intrinsics"],
            pose=camera_pose
        ).to(device=self.device)

        self.world_model.add_camera_frame(data_camera, "world")  # "world" layer
        self.world_model.process_camera_frames("world", False)
        torch.cuda.synchronize()
        self.world_model.update_blox_hashes()

    def save_tsdf(self, filename='tsdf_map.nvb'):
        self.world_model.save_layer("world", filename)
        self.get_logger().info(f"Saved TSDF map to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseToBaseLinkTSDF()

    try:
        for _ in range(500):  # collect 500 frames
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_tsdf_from_camera()
    except KeyboardInterrupt:
        pass

    # Save TSDF for offline use
    node.save_tsdf("tsdf_base_link.nvb")

    # Stop Realsense device
    node.realsense.stop_device()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
