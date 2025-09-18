#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script for doing a scanning trajectory with Motion Stack

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from typing import List, Callable, Dict
import motion_stack.ros2.ros2_asyncio.ros2_asyncio as rao
from motion_stack.ros2.utils.executor import error_catcher
from motion_stack.api.ros2.joint_api import JointHandler, JointSyncerRos
from ament_index_python.packages import get_package_share_directory


class TrajectoryExecution(Node):
    def __init__(self, trajectory_execution_config=None, trajectory_file=None, csv_delimiter= ' '):
        super().__init__("trajectory_execution")

        # Get package path
        package_share_path = get_package_share_directory("motion_planning_trajectory_execution")
        print("PKG DONE")

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
        traj_file_name = self.traj_exec_cfg["trajectory_file"]

        if trajectory_execution_config is None:
            traj_path = os.path.join(package_share_path, "trajectories", traj_file_name)
        else:
            traj_path = os.path.join(package_share_path, "trajectories", trajectory_file)

        try:
            self.trajectory = np.loadtxt(traj_path)
        except FileNotFoundError:
            raise SystemExit("Trajectory file not found!")
        except Exception as e:
            raise SystemExit("Error loading trajectory file: {e}")

        # Trajectory tracking
        self.waypoints = self.trajectory.shape[0]
        self.current_waypoint = 0  # row of trajectory file 

        # Create timer
        self.done_once = False
        self.period = 1.0 / self.traj_exec_cfg["control_plugin_frequency"]
        self.timer = self.create_timer(self.period, self.on_timer)

        # Orchestration of steps
        self.steps: List[Callable[[], None]] = []  # Queue of steps (similar to state machine)

        # Get controlled leg ID and joints
        self.LIMBS = self.traj_exec_cfg["controlled_robot_IDs"] 
        self.LEG_JOINTS: List[str] = self.traj_exec_cfg["controlled_joints"]
        self.SET_JOINT_POSITION: List[float] = self.traj_exec_cfg["defined_joint_position"]

        # Motion Stack
        self.joint_handlers: JointHandler = [JointHandler(self, l) for l in self.LIMBS]
        self.joint_syncer: JointSyncerRos = JointSyncerRos(self.joint_handlers) # coordinate joints and safety measures

        self.get_logger().info("TrajectoryExecution node initialized.")


    @error_catcher
    def on_timer(self):
        """ Start timer and steps """
        try: 
            # Execute once at startup
            if not self.done_once: 
                self.done_once = True
                self.steps = [
                    self.step_wait_joints_ready, 
                    # self.step_zero_position,
                    # self.step_trajectory_execution, 
                    self.step_go_to_set_position,
                    self.step_finished,
                ]
                self.run_next_step()
                self.get_logger().info("Startup done. Sequence of steps queued.")

            # Regularly execute Motion Stack JointSyncer
            self.joint_syncer.execute()

        except Exception as e:
            self.get_logger().error(f"Runtime error in timer loop: {e}")


    def run_next_step(self):
        """ Move to the next step of the list """
        if not self.steps:
            return
        step = self.steps.pop(0)
        step()  # starts the step; the step must call run_next_step() when done


    def step_wait_joints_ready(self):
        """Wait for the future of all joints of all limbs to be ready """
        self.get_logger().info("Waiting for joints...")

        # Futures of each individual limb (one joint_handler per limb)
        limbs_futures = [jh.ready for jh in self.joint_handlers] 

        # Attach a callback (log) to each jh.ready's future
        for jh in self.joint_handlers:
            jh.ready.add_done_callback(
                lambda _f, limb=jh.limb_number: self.get_logger().info(f"Joints ready: limb {limb}")
            )

        # Create a single future that completes when all limbs_futures are complete
        all_ready_future = rao.ensure_future(self, rao.gather(self, *limbs_futures))
        
        # When the "all ready_future" future completes, log once and go to next step
        all_ready_future.add_done_callback(lambda _f: self.after_future("ALL joints of ALL limbs ready", all_ready_future))


    def step_trajectory_execution(self):
        """Feed a trajectory and complete each waypoint"""
        self.get_logger().info(f"Executing a trajectory with {self.waypoints} waypoints...")
        
        # Empty trajectory
        if self.trajectory is None:
            self.run_next_step()
            return

        # If trajectory is completed move to next step
        if self.current_waypoint >= self.waypoints:
            self.trajectory = None
            self.after_future("Trajectory DONE", self.waypoint_reached_future)
            return
        
        print(self.trajectory[self.current_waypoint])
        
        # Send waypoint's joint states 
        target_joint_state: Dict[str, float] = {
            joint: value for joint, value in zip(self.LEG_JOINTS, self.trajectory[self.current_waypoint])
        }
        # self.get_logger().info(target_joint_state)

        self.waypoint_reached_future = self.joint_syncer.lerp(target_joint_state)

        # Add callback for when waypoint is reached 
        def waypoint_done(_):
            
            self.current_waypoint += 1
            _ = self.waypoint_reached_future.result() # Raise if error
            self.get_logger().info(f"Waypoint {self.waypoints} reached...")
            self.step_trajectory_execution()
        
        self.waypoint_reached_future.add_done_callback(waypoint_done)


    def step_zero_position(self):
        self.get_logger().info("Sending all joints to zero...")
        target_joint_state: Dict[str, float] = {
            joint: 0.0 for joint in self.LEG_JOINTS
        }
    
        zero_reached_future = self.joint_syncer.lerp(target_joint_state)
        zero_reached_future.add_done_callback(lambda f: self.after_future("Angles at defined position", f))


    def step_go_to_set_position(self):
        self.get_logger().info("Sending all joints to defined configuration...")
        target_joint_state: Dict[str, float] = {
            joint: value for (joint, value) in zip(self.LEG_JOINTS, self.SET_JOINT_POSITION)
        }
    
        position_reached_future = self.joint_syncer.lerp(target_joint_state)
        position_reached_future.add_done_callback(lambda f: self.after_future("Angles at defined position", f))


    def step_finished(self):
        self.get_logger().info("Sequence finished ✅")


    def after_future(self, msg: str, fut):
        try:
            _ = fut.result()  # raise if error
            self.get_logger().info(msg)
            self.run_next_step()
        except Exception as e:
            self.get_logger().error(f"{msg} FAILED: {e}")
