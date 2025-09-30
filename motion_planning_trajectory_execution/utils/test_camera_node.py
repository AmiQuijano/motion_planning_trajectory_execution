#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MOTION STACK MUST BE RAN FIRST

import rclpy
import yaml
import cv2
import torch
import os

from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory


from motion_planning_trajectory_execution.utils.realsense_dataset import RealsenseDataloader
# from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

class CameraRecorder(Node):
    def __init__(self, camera_recorder_config=None):
        super().__init__("camera_recorder")

        # Get package path
        package_share_path = get_package_share_directory("motion_planning_trajectory_execution")
        print("PKG DONE")

        # Load YAML config file
        if camera_recorder_config is None:
            cfg_path = os.path.join(package_share_path, "configs", "camera_recorder_config.yaml")
        else:
            cfg_path = os.path.join(package_share_path, "configs", camera_recorder_config)

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.camera_recorder_cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise SystemExit("Camera recording config file not found!")
        except yaml.YAMLError as e:
            raise SystemExit(f"YAML syntax error: {e}")

        # Get parameters from config file
        self.frame_a = self.camera_recorder_cfg["frame_a"]
        self.frame_b = self.camera_recorder_cfg["frame_b"]
        self.save_rate = self.camera_recorder_cfg["save_rate"]
        self.max_frames = self.camera_recorder_cfg["max_frames"]
        clipping_distance = self.camera_recorder_cfg["clipping_distance"]
        file_name = self.camera_recorder_cfg["save_file"]

        # Start Realsense
        self.realsense_data = RealsenseDataloader(clipping_distance_m=clipping_distance, serial_number="233522074308")

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize variables
        self.frames = []        # List of frames' data
        self.frames_saved = 0
        
        # Path for saving frames data
        # self.save_path = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", file_name)
        self.save_path = os.path.join(package_share_path, "camera_data", file_name)


    def wait_for_tf(self):
        """Block process until TFs are available"""
        self.get_logger().info(f"Waiting for TF {self.frame_a} --> {self.frame_b} ...")
        while rclpy.ok():
            try:
                tf = self.tf_buffer.lookup_transform(self.frame_b, self.frame_a, rclpy.time.Time())
                self.get_logger().info("TF found, starting recording!")
                return
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

    
    def visualize_color_depth(self, color_np, depth_np):
        """Show depth and color frames using OpenCV"""
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_np, alpha=100), cv2.COLORMAP_VIRIDIS
        )
        stacked = cv2.hconcat([color_np, depth_colormap])
        cv2.imshow("Camera Recording: Color + Depth", stacked)
        key = cv2.waitKey(1)
        return key

    # def visualize_color_depth(self, color_np, depth_np):
    #     """Show depth and color frames using OpenCV"""

    #     # Ensure color is uint8
    #     if color_np.dtype != 'uint8':
    #         color_disp = (color_np * 255).astype('uint8')
    #     else:
    #         color_disp = color_np.copy()

    #     # Convert RGB → BGR for OpenCV
    #     color_disp = cv2.cvtColor(color_disp, cv2.COLOR_RGB2BGR)

    #     # Depth colormap
    #     depth_colormap = cv2.applyColorMap(
    #         cv2.convertScaleAbs(depth_np, alpha=100), cv2.COLORMAP_VIRIDIS
    #     )

    #     # Stack side by side
    #     stacked = cv2.hconcat([color_disp, depth_colormap])
    #     cv2.imshow("Camera Recording: Color + Depth", stacked)
    #     key = cv2.waitKey(1)
    #     return key


    def run(self):
        """Main run"""
        # 1. Wait for TFs to be available
        self.wait_for_tf()

        self.get_logger().info("Waiting for RealSense frames...")
        frame_counter = 1 
        while rclpy.ok():
            try:
                # 2. Get Realsense depth, color and instrinsics
                data = self.realsense_data.get_data()
                self.get_logger().info(
                    f"Got frame {frame_counter}: Depth {data['depth'].shape}, Intrinsics {data['intrinsics'].shape}"
                )
                depth_tensor = data["depth"]
                depth_np = data["raw_depth"]
                rgb_np = data["raw_rgb"].copy()
                rgba_tensor = data["rgba"]
                rbga_nvblox_tensor = data["rgba_nvblox"]
                intrinsics_tensor = data["intrinsics"]

                # 3. Get latest camera --> base_link transform 
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    self.frame_b, self.frame_a, rclpy.time.Time()
                )

                # 4. Allocate camera pose
                position_tensor = torch.tensor([tf.transform.translation.x,
                                                tf.transform.translation.y,
                                                tf.transform.translation.z],
                                                dtype=torch.float32)
                quaternion_tensor = torch.tensor([tf.transform.rotation.w,
                                                  tf.transform.rotation.x,
                                                  tf.transform.rotation.y,
                                                  tf.transform.rotation.z],
                                                  dtype=torch.float32)
                
                print(f"Got camera position and quaternion for frame {frame_counter}")

                frame_counter +=1

                # 5. Visualize color and depth frames
                key = self.visualize_color_depth(rgb_np, depth_np)
                if key & 0xFF == ord("q") or key == 27:  # q or ESC to close and stop recording
                    self.get_logger().info("Recording interrupted by user.")
                    break

            except StopIteration:
                # time.sleep(0.01)
                continue
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # Stop recording once the max number of frames has been saved
            if self.frames_saved == self.max_frames:
                break
            
            # 6. Save frames data in list
            if frame_counter % self.save_rate == 0:
                self.frames.append({
                    # "depth": depth,
                    # "intrinsics": intrinsics,
                    # "position": position,
                    # "quaternion": quaternion 
                    "depth_tensor": depth_tensor,
                    "depth_np": depth_np,
                    "rgb_np": rgb_np,
                    "rgba_tensor": rgba_tensor,
                    "rbga_nvblox_tensor": rbga_nvblox_tensor,
                    "intrinsics_tensor": intrinsics_tensor,
                    "position_tensor": position_tensor,
                    "quaternion_tensor": quaternion_tensor
                })
                self.frames_saved += 1
                self.get_logger().info(f"Recorded frame {frame_counter}, {self.frames_saved}/{self.max_frames}")
                
        # 7. Export frames data list to file
        torch.save(self.frames, self.save_path)
        self.get_logger().info(f"Saved {len(self.frames)} frames to {self.save_path}")

        self.realsense_data.stop_device()


def main(args=None):
    rclpy.init(args=args)
    node = CameraRecorder()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

