#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script that takes camera intrinsics and depth through pyrealsense2's
# torch wrapper 'nvblox_torch' and pose from ROS2 tf topics.
# This data is saved into a file for later use


import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

import cv2
import torch
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader
from curobo.types.math import Pose
import time


SAVE_PATH = "camera_recorded_data2.pt"  # file to save all frames
NUM_FRAMES = 10  # how many frames to record

# Consider that 
# NUM_FRAMES = 30fps * recording_time (in sec)
# Thus for 10s recording, NUM_FRAMES = 300

class CameraRecorder(Node):
    def __init__(self):
        super().__init__("camera_recorder")

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # RealSense loader
        self.clipping_distance = 0.8 # [m]
        self.realsense_data = RealsenseDataloader(self.clipping_distance)
        # time.sleep(5)

        self.frames = []

        # Transform of <a> with respect to <b>
        self.frame_a = "camera_color_optical_frame"
        self.frame_b = "base_link"


    def wait_for_tf_and_frame(self):
        """Wait until both a valid TF and camera frame are available"""
        self.get_logger().info(f"Waiting for TF {self.frame_a} -> {self.frame_b} and a valid camera frame...")

        while rclpy.ok():
            tf_valid = False
            frame_valid = False

            # Check TF
            try:
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    self.frame_b, self.frame_a, rclpy.time.Time()
                )
                tf_valid = True
                print("TFS FOUND!!")
            except Exception:
                print("NO TFSSSS")
                pass

            # Check camera frame
            try:
                data = self.realsense_data.get_data()
                if "depth" in data and data["depth"] is not None:
                    frame_valid = True
                    print("FRAMES FOUND!!")
            except StopIteration:
                print("NO DEPTH OR INTRINSICS")
                time.sleep(0.01)
                continue

            if tf_valid and frame_valid:
                self.get_logger().info("TF and camera frame are ready!")
                return #data, tf  # return both immediately

            rclpy.spin_once(self, timeout_sec=0.01)  # allow ROS to process callbacks


    def wait_for_tf(self):
        """Block process until TFs are available"""
        self.get_logger().info(f"Waiting for TF {self.frame_a} --> {self.frame_b} ...")
        while rclpy.ok():
            try:
                tf = self.tf_buffer.lookup_transform(self.frame_b, self.frame_a, rclpy.time.Time())
                self.get_logger().info("TF found, starting recording!")
                return tf
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)

    
    def get_camera_pose(self):
        """Wait until a TF is available and return it as a Pose object"""
        while rclpy.ok():
            try:
                tf: TransformStamped = self.tf_buffer.lookup_transform(
                    self.frame_b, self.frame_a, rclpy.time.Time()
                )
                position = torch.tensor([tf.transform.translation.x,
                                         tf.transform.translation.y,
                                         tf.transform.translation.z],
                                        dtype=torch.float32)
                quaternion = torch.tensor([tf.transform.rotation.x,
                                           tf.transform.rotation.y,
                                           tf.transform.rotation.z,
                                           tf.transform.rotation.w],
                                          dtype=torch.float32)
                return Pose(position=position, quaternion=quaternion)
            except Exception:
                # Wait briefly before retrying
                rclpy.spin_once(self, timeout_sec=0.01)
    

    def get_realsense_data(self):
        """Fetch depth, intrinsics and color frames from Realsense"""
        data = self.realsense_data.get_data()

        depth = data["depth"]            # torch.Tensor [480, 640]
        color = data["rgba_nvblox"]      # torch.Tensor [480, 640, 3]
        intrinsics = data["intrinsics"]  # torch.Tensor [3,3]

        return depth, color, intrinsics 
    

    def visualize_depth_color(self, depth, color):
        """Show depth and color using OpenCV"""
        color_np = color.cpu().numpy().astype("uint8")
        depth_np = depth.cpu().numpy()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_np, alpha=100), cv2.COLORMAP_VIRIDIS
        )
        stacked = cv2.hconcat([color_np, depth_colormap])
        cv2.imshow("Camera Recording: Color + Depth", stacked)
        key = cv2.waitKey(1)
        return key
    

    # def wait_for_camera(self):
    #     self.get_logger().info("Waiting for RealSense frames...")
    #     while rclpy.ok():
    #         data = self.realsense_data.get_data()
    #         if data and "depth" in data and data["depth"] is not None:
    #             self.get_logger().info("Camera frames available!")
    #             return
    #         rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_camera(self):
        """Wait until a valid RealSense frame is available"""
        self.get_logger().info("Waiting for RealSense frames...")
        while rclpy.ok():
            try:
                data = self.realsense_data.get_data()
                if data and "depth" in data and data["depth"] is not None and \
                "intrinsics" in data and data["intrinsics"] is not None:
                    return data
            except StopIteration:
                # The library raised StopIteration because a frame was not ready yet
                pass
            except Exception as e:
                self.get_logger().warn(f"Error getting camera frame: {e}")

            time.sleep(0.01)  # small wait before trying again

    
    def run(self):
        """Main run"""
        # 1. Wait until TFs exist
        # self.wait_for_tf()
        # self.wait_for_tf_and_frame()
        data = self.wait_for_camera()
        print("CAMERA DATA OBTAINED")

        # i = 0
        # while i < NUM_FRAMES and rclpy.ok():
        #     # 2. Get frame information
        #     try: 
        #         depth, color, intrinsics = self.get_realsense_data()
        #         print("CAMERA DATA OBTAINED")
        #     except Exception as e:
        #         self.get_logger().warn(f"Skipping frame {i}, error: {e}")
        #         continue
        #     i += 1


def main(args=None):
    rclpy.init(args=args)
    recorder = CameraRecorder()
    recorder.run()
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
