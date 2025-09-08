#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# CuRobo imports (from omni_python env)
from curobo.types.math import Pose
import torch


class HelloCurobo(Node):
    def __init__(self):
        super().__init__("hello_curobo")

        # Just create a Pose with CuRobo and log it
        t = torch.tensor([0.1, 0.2, 0.3], device="cpu")
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cpu")
        pose = Pose(position=t, quaternion=q)

        self.get_logger().info(f"Created CuRobo Pose: {pose}")


def main(args=None):
    rclpy.init(args=args)
    node = HelloCurobo()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
