#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script that reads .csv with manipulator trajectory and uses 'Motion Stack' APIs to execute the trajectory 

import yaml
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from typing import Awaitable, List, Tuple
from motion_stack.api.ros2.joint_api import JointHandler, JointSyncerRos
from ament_index_python.packages import get_package_share_directory


class TrajectoryExecution(Node):
    def __init__(self, trajectory_execution_config=None, trajectory_file=None, csv_delimiter= ' '):
        super().__init__("trajectory_execution_old")

        # Get package path
        package_share_path = get_package_share_directory("motion_planning_trajectory_execution")

        # Load YAML config file
        if trajectory_execution_config is None:
            cfg_path = os.path.join(package_share_path, "configs", "trajectory_execution_config.yaml")
        else:
            cfg_path = os.path.join(package_share_path, "configs", trajectory_execution_config)

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.traj_exec_cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise SystemExit("Trajectory config file not found!")
        except yaml.YAMLError as e:
            raise SystemExit(f"YAML syntax error: {e}")
        
        # Load CSV trajectory file
        if trajectory_execution_config is None:
            traj_path = os.path.join(package_share_path, "trajectories", "trajectory.csv")
        else:
            traj_path = os.path.join(package_share_path, "trajectories", trajectory_file)

        try:
            self.trajectory = np.loadtxt(traj_path)
        except FileNotFoundError:
            raise SystemExit("Trajectory file not found!")
        except Exception as e:
            raise SystemExit("Error loading trajectory file: {e}")

        # Get controlled leg ID and joints
        LEG_ID = self.traj_exec_cfg["controlled_robot_IDs"][0]  # Realman ID = 75
        self.LEG_JOINTS: List[str] = self.traj_exec_cfg["controlled_joints"]

        # Motion Stack
        self.joint_handler: JointHandler = JointHandler(self, LEG_ID)
        self.joint_syncer: JointSyncerRos = JointSyncerRos([self.joint_handler]) # coordinate joints and safety measures

        # Setup robot futures
        self.robots_futures = self.robots_setup()
        self.robots_ready: bool = False

        # Trajectory tracking
        self.current_idx = 0  # row index of trajectory file 
        self.current_future: Future = None

        # Loop rate and timer
        self._period = 1.0 / self.traj_exec_cfg["control_plugin_frequency"]
        self._timer = self.create_timer(self._period, self._on_timer)

        self.get_logger().info("TrajectoryExecution node initialized.")

    def robots_setup(self) -> List[Tuple[Awaitable, Awaitable]]:
        """Initiates the robots setup procedure in Motion Stack API and returns list of Future's for the robots."""
        self.get_logger().info(f"Trying to setup the following joints: {self.LEG_JOINTS}")
        leg_setup = self.joint_handler.ready_up(set(self.LEG_JOINTS))
        self.get_logger().info(f"Setup done")
        return [leg_setup]
    
    def check_robot_ready(self):
        """Checks if all joints on the robot are ready.
        If yes, then sets the robots_ready flag to True and allows for trajectory execution.
        """
        # Shortcut
        if self.robots_ready:
            return

        # ALL robots need to be ready
        if all(future[0].done() for future in self.robots_futures):
            self.robots_ready = True
            self.get_logger().info(f"All robots are ready.")
            return

        pass

    def send_joint_cmd(self, state: List[float]) -> Future:
        """Send joint state commands using 'JointHandler's and 'JointSyncer's."""
        joint_state= {
            joint: state for (joint, state) in zip(self.LEG_JOINTS, state)}
        
        # Send joint state
        fut = self.joint_syncer.lerp(joint_state)
        return fut

    def _on_timer(self):
        try:
            self.check_robot_ready()
            if not self.check_robot_ready():
                self.get_logger().info("Waiting for robot to be ready...")
                return
            
            # If a previous joint command is still executing, wait
            if self.current_future is not None and not self.current_future.done():
                return
            
            # Trajectory finished
            if self.current_idx >= len(self.trajectory):
                self.get_logger().info("Trajectory execution finished!")
                self._timer.cancel()
                return
            
            # Send next joint state
            joint_state = self.trajectory[self.current_idx].tolist()
            self.current_future = self.send_joint_cmd(joint_state)
            self.current_idx += 1

            self.joint_syncer.execute()

        except Exception as e:
            self.get_logger().error(f"Runtime error in timer loop: {e}")

    def main():
        rclpy.init()
        node = None
        try:
            node = TrajectoryExecution()
            rclpy.spin(node)
        except SystemExit as e:
            # Already logged; exit cleanly
            pass
        except Exception as e:
            # Unexpected exception
            if node:
                node.get_logger().error(f"Unhandled exception: {e}")
            else:
                print(f"Unhandled exception before node init: {e}")
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()