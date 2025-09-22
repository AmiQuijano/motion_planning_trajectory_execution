#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

import torch
import cv2

from motion_planning_trajectory_execution.utils.realsense_dataset import RealsenseDataloader
from motion_planning_trajectory_execution.action import CameraRecord

class CameraRecordActionServer(Node):
    def __init__(self):
        super().__init__('camera_record_action_server')

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        
        self.realsense_data = RealsenseDataloader(clipping_distance_m=0.5)

        self.frame_a = "camera_color_optical_frame"
        self.frame_b = "base_link"
        self.frames = []
        self.save_rate = 5
        self.frames_saved = 0

        self._action_server = ActionServer(
            self,
            ScanEnvironment,
            'scan_environment',
            self.execute_callback
        )

    def wait_for_tf(self):
        self.get_logger().info(f"Waiting for TF {self.frame_a} -> {self.frame_b} ...")
        while rclpy.ok():
            try:
                tf = self.tf_buffer.lookup_transform(self.frame_b, self.frame_a, rclpy.time.Time())
                self.get_logger().info("TF found!")
                return
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)

    def visualize_color_depth(self, color_np, depth_np):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=100), cv2.COLORMAP_VIRIDIS)
        stacked = cv2.hconcat([color_np, depth_colormap])
        cv2.imshow("Camera Recording: Color + Depth", stacked)
        key = cv2.waitKey(1)
        return key

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        save_path = goal.save_path
        num_frames = goal.num_frames

        self.get_logger().info(f"Recording {num_frames} frames to {save_path}")
        self.wait_for_tf()

        frame_counter = 1
        self.frames = []
        self.frames_saved = 0

        while rclpy.ok() and self.frames_saved < num_frames:
            try:
                data = self.realsense_data.get_data()
                depth = data["depth"]
                depth_np = data["raw_depth"]
                color_np = data["raw_rgb"]
                intrinsics = data["intrinsics"]

                tf: TransformStamped = self.tf_buffer.lookup_transform(self.frame_b, self.frame_a, rclpy.time.Time())
                position = torch.tensor([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z], dtype=torch.float32)
                quaternion = torch.tensor([tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z], dtype=torch.float32)

                # Visualize frames
                key = self.visualize_color_depth(color_np, depth_np)
                if key & 0xFF == ord("q") or key == 27:
                    self.get_logger().info("Recording interrupted by user.")
                    break

            except StopIteration:
                continue

            rclpy.spin_once(self, timeout_sec=0.01)

            # Save frame every `save_rate`
            if frame_counter % self.save_rate == 0:
                self.frames.append({
                    "depth": depth,
                    "intrinsics": intrinsics,
                    "position": position,
                    "quaternion": quaternion
                })
                self.frames_saved += 1
                self.get_logger().info(f"Recorded frame {frame_counter}, {self.frames_saved}/{num_frames}")

            frame_counter += 1

        # Save all frames to disk
        torch.save(self.frames, save_path)
        self.get_logger().info(f"Saved {len(self.frames)} frames to {save_path}")
        self.realsense_data.stop_device()

        goal_handle.succeed()
        result = ScanEnvironment.Result()
        result.success = True
        result.save_path = save_path
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CameraActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
