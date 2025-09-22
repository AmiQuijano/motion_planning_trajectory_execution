#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script that reads .csv with manipulator trajectory and uses 'Motion Stack' APIs to execute the trajectory 

import yaml
import os
import copy
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from typing import List, Callable, Dict
from ament_index_python.packages import get_package_share_directory
import motion_stack.ros2.ros2_asyncio.ros2_asyncio as rao
from motion_stack.ros2.utils.executor import error_catcher
from motion_stack.api.ros2.joint_api import JointHandler, JointSyncerRos
from motion_stack.core.utils.joint_state import JState


class TrajectoryExecution(Node):
    def __init__(self, trajectory_execution_config=None, trajectory_file=None):
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
            self.csv_trajectory = np.loadtxt(traj_path)
            self.csv_waypoints = self.csv_trajectory.shape[0]
        except FileNotFoundError:
            raise SystemExit("Trajectory file not found!")
        except Exception as e:
            raise SystemExit("Error loading trajectory file: {e}")

        # Trajectory tracking
        # self.waypoints = self.csv_trajectory.shape[0]
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
        self.SET_JOINT_STATES: List[List[float]] = self.traj_exec_cfg["defined_joint_states"]
        self.STEPS = self.traj_exec_cfg["n_steps"]

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
                    lambda: self.step_set_state(self.SET_JOINT_STATES[0]),  
                    # [
                    #   ['step set state', arg1],
                    #   ['step set state steps', arg1, arg2]
                    # ]
                    lambda: self.step_set_state_steps(self.SET_JOINT_STATES[1], self.STEPS),  
                    # self.step_zero_position,   
                    # lambda: self.step_trajectory_execution(self.csv_trajectory, self.csv_waypoints), 
                    lambda: self.step_set_state(self.SET_JOINT_STATES[2]),  
                    lambda: self.step_set_state_steps(self.SET_JOINT_STATES[3], self.STEPS),
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


    def step_zero_position(self):
        """Send all joints to zero position"""
        self.get_logger().info("Sending all joints to zero...")
        target_joint_state: Dict[str, float] = {
            joint: 0.0 for joint in self.LEG_JOINTS
        }
    
        zero_reached_future = self.joint_syncer.lerp(target_joint_state)
        zero_reached_future.add_done_callback(lambda f: self.after_future("Angles at zero position", f))


    def step_set_state(self, desired_state):
        """Go to a desired joint state as fast as possible"""
        self.get_logger().info("Sending all joints to defined state...")
        target_joint_state: Dict[str, float] = {
            joint: value for (joint, value) in zip(self.LEG_JOINTS, desired_state)
        }
    
        state_reached_future = self.joint_syncer.lerp(target_joint_state)
        state_reached_future.add_done_callback(lambda f: self.after_future("Angles at defined position", f))

    
    def step_set_state_steps(self, desired_state, n_steps):
        """Go to a desired joint state in n steps (the larger n the slower the motion, and viceversa)"""
        self.get_logger().info("Sending all joints to defined state...")

        states = self.get_states()
        leg_state = [states[k].position for k in self.LEG_JOINTS] # Get current joint states of leg
        print(leg_state)

        if None in leg_state:
            self.get_logger().info(f"'None' in useful states")
            self.run_next_step()
            return
        
        start_state = np.array(leg_state)
        end_state = np.array(desired_state)

        trajectory = np.linspace(start_state, end_state, num=n_steps)
        n_waypoints = trajectory.shape[0]
        self.current_waypoint = 0

        self.step_trajectory_execution(trajectory, n_waypoints)
        

    def step_trajectory_execution(self, trajectory, n_waypoints):
        """Follow a trajectory from a .csv file by completing each waypoint"""
        self.get_logger().info(f"Executing a trajectory with {n_waypoints} waypoints...")

        # Empty trajectory
        if trajectory is None:
            self.run_next_step()
            return

        # If trajectory is completed move to next step
        if self.current_waypoint >= n_waypoints:
            trajectory = None
            self.current_waypoint = 0
            self.after_future("Trajectory DONE", self.waypoint_reached_future)
            return
        
        print(trajectory[self.current_waypoint])
        
        # Send waypoint's joint states 
        target_joint_state: Dict[str, float] = {
            joint: value for joint, value in zip(self.LEG_JOINTS, trajectory[self.current_waypoint])
        }
        # self.get_logger().info(target_joint_state)

        self.waypoint_reached_future = self.joint_syncer.lerp(target_joint_state)

        # Add callback for when waypoint is reached 
        def waypoint_done(_):
            
            self.current_waypoint += 1
            _ = self.waypoint_reached_future.result() # Raise if error
            self.get_logger().info(f"Waypoint {self.current_waypoint} reached...")
            self.step_trajectory_execution(trajectory, n_waypoints)
        
        self.waypoint_reached_future.add_done_callback(waypoint_done)


    def step_finished(self):
        """Confirm all steps are done"""
        self.get_logger().info("Sequence finished ✅")


    def after_future(self, msg: str, fut):
        try:
            _ = fut.result()  # raise if error
            self.get_logger().info(msg)
            self.run_next_step()
        except Exception as e:
            self.get_logger().error(f"{msg} FAILED: {e}")

    
    def get_states(self) -> Dict[str, JState]:
        """
        Get a dictionary with joint state names as keys and joint states as values.

        Returns:
            out (Dict[str, JState]): Dictionary of joint states with joint names as keys
        """
        out = {}
        for jh in self.joint_handlers:
            out.update({v.name: v for v in jh.states})
        return copy.deepcopy(out)
