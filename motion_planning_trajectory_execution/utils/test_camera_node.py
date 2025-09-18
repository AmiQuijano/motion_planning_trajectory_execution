#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

import cv2
import torch
import os

from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader
from curobo.types.math import Pose


SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", "camera_recorded_data3.pt")  # file to save all frames
NUM_FRAMES = 300  # how many frames to record

class RealsenseMinimal(Node):
    def __init__(self):
        super().__init__("realsense_minimal")

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.realsense_data = RealsenseDataloader(clipping_distance_m=0.5)

        # Transform of <a> with respect to <b>
        self.frame_a = "camera_color_optical_frame"
        self.frame_b = "base_link"

        # Frames 
        self.frames = []        # List of frames data
        self.save_rate = 5     # Save every n frames
        self.frames_saved = 0


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
                depth = data["depth"]
                depth_np = data["raw_depth"]
                color_np = data["raw_rgb"]
                intrinsics = data["intrinsics"]

                # 3. Get latest camera --> base_link transform 
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    self.frame_b, self.frame_a, rclpy.time.Time()
                )

                # print(tf.transform.translation)
                # print(tf.transform.rotation)

                # 4. Allocate camera pose
                position = torch.tensor([tf.transform.translation.x,
                                         tf.transform.translation.y,
                                         tf.transform.translation.z],
                                        dtype=torch.float32)
                quaternion = torch.tensor([tf.transform.rotation.x,
                                           tf.transform.rotation.y,
                                           tf.transform.rotation.z,
                                           tf.transform.rotation.w],
                                          dtype=torch.float32)
                
                # print(position.dtype)
                # print(quaternion.dtype)
                
                # camera_pose = Pose(position=position, quaternion=quaternion)
                print(f"Got camera position and quaternion for frame {frame_counter}")

                frame_counter +=1

                # 5. Visualize color and depth frames
                key = self.visualize_color_depth(color_np, depth_np)
                if key & 0xFF == ord("q") or key == 27:  # q or ESC to close and stop recording
                    self.get_logger().info("Recording interrupted by user.")
                    break

            except StopIteration:
                # time.sleep(0.01)
                continue
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # Stop recording once the max number of frames has been saved
            if self.frames_saved == NUM_FRAMES:
                break
            
            # 6. Save frames data in list
            if frame_counter % self.save_rate == 0:
                self.frames.append({
                    "depth": depth,
                    "intrinsics": intrinsics,
                    # "pose": camera_pose
                    "position": position,
                    "quaternion": quaternion 
                })
                self.frames_saved += 1
                self.get_logger().info(f"Recorded frame {frame_counter}, {self.frames_saved}/{NUM_FRAMES}")
                
        # 7. Export frames data list to file
        torch.save(self.frames, SAVE_PATH)
        self.get_logger().info(f"Saved {len(self.frames)} frames to {SAVE_PATH}")

        self.realsense_data.stop_device()


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseMinimal()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

