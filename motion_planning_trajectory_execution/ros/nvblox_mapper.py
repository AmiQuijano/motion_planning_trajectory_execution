## NOT TESTED, MIGHT NOT WORK
## NOT USING IT ANYMORE

import rclpy
import yaml
import torch
import tf2_ros
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg

from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader
from curobo.types.camera import CameraObservation, Pose
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig
from curobo.wrap.model.robot_world import RobotWorld

class NvbloxMapper(Node):
    def __init__(self):
        super().__init__("nvblox_mapper")

        # Parameters
        self.reference_frame = "base_link"                # fixed world frame
        self.camera_frame = "camera_depth_optical_frame"    # depth frame for intrinsics
        self.clipping_distance = 0.8                        # max depth (meters)
        self.voxel_size = 0.02                              # ize of cubic voxels (meters)
        self.edge_clip_ratio = 0.05                         # fraction of edges to zero
        self.device = 'cuda:0'

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # RealSense loader
        self.realsense = RealsenseDataloader(clipping_distance_m=self.clipping_distance)        
        
        # NVBlox world configuration
        # self.tensor_args = TensorDeviceType(device=self.device)
        self.tensor_args = TensorDeviceType()
        self.world_cfg = WorldConfig.from_dict({
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "tsdf",
                    "voxel_size": self.voxel_size,
                }
            }
        })
        print("BEFORE")
        self.world_model = RobotWorld(self.world_cfg, self.tensor_args).world_collision
        print("AFTER")

         # ROS2 PointCloud2 publisher for RViz
        self.pcd_pub = self.create_publisher(PointCloud2, 'nvblox_voxels', 1)

        self.get_logger().info("NvbloxMapper Node Initialized")

    def get_camera_pose(self):
        """Get camera pose in reference_frame"""
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.reference_frame, self.camera_frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            return Pose(
                position=self.tensor_args.to_device([t.x, t.y, t.z]),
                quaternion=self.tensor_args.to_device([q.w, q.x, q.y, q.z])
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None


    def clip_depth_image(self, depth_image):
        """Clip depth edges and max depth"""
        h, w = depth_image.shape

        # Clip edges
        h_clip = int(self.edge_clip_ratio * h)
        w_clip = int(self.edge_clip_ratio * w)
        depth_image[:h_clip, :] = 0.0
        depth_image[-h_clip:, :] = 0.0
        depth_image[:, :w_clip] = 0.0
        depth_image[:, -w_clip:] = 0.0

        # Clip max depth
        depth_image[depth_image > self.clipping_distance] = 0.0

        return depth_image


    def step(self):
        # 1. Get camera data
        data = self.realsense.get_data()

        # Clip depth before integration

        depth_tensor = data["depth"]
        depth_tensor = self.clip_depth_image(depth_tensor)
        data["depth"] = depth_tensor

        # 2. Get camera pose
        pose = self.get_camera_pose()
        if pose is None:
            self.get_logger().info("Pose is NONE")
            return

        # 3. Integrate depth into TSDF
        obs = CameraObservation(
            depth_image=data["depth"],
            intrinsics=data["intrinsics"],
            pose=pose,
        ).to(device=self.tensor_args.device)

        # Add new camera observation (no decay → map only grows)
        self.world_model.add_camera_frame(obs, "world")
        self.world_model.process_camera_frames("world", False)
        torch.cuda.synchronize()
        self.world_model.update_blox_hashes()

        # 5. Publish voxels to RViz
        self.publish_voxels()


    def publish_voxels(self):
        """Publish current TSDF voxels as PointCloud2 for RViz"""
        bounding = Cuboid("world_bbox", dims=[10, 10, 3.0], pose=[0, 0, 0, 1, 0, 0, 0])
        voxels = self.world_model.get_voxels_in_bounding_box(bounding, self.voxel_size)
        if voxels.shape[0] == 0:
            return

        voxels_np = voxels.cpu().numpy()[:, :3]

        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.reference_frame

        msg = pc2.create_cloud_xyz32(header, voxels_np.tolist())
        self.pcd_pub.publish(msg)


    def save_all(self):
        # Save mesh (PLY/OBJ/STL supported)
        mesh = self.world_model.get_mesh("world")
        mesh.export("map_mesh.obj")

        # Save WorldConfig YAML (reloadable in CuRobo)
        yaml_dict = self.world_cfg.to_dict()
        with open("map_worldconfig.yml", "w") as f:
            yaml.safe_dump(yaml_dict, f)

        # Save full TSDF voxel layer
        self.world_model.save_layer("world", "map_layer.tsdf")