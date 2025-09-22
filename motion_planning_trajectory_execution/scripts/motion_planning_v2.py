#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script that loads camera tensor data previously recorded (depth, intrinsics, pose) to
# build an nvblox map of obstacles and perform collision-avoidance motion planning

try:
    import isaacsim
except ImportError:
    pass

import torch
import os
import cv2

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)

parser.add_argument("--robot", type=str, default="rm75_6f_realsense.yml", help="robot configuration to load")

parser.add_argument(
    "--use-debug-draw",
    action="store_true",
    help="When True, sets robot in static mode",
    default=False,
)
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

import numpy as np
import trimesh
import torch
import carb
from matplotlib import cm
# from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

simulation_app.update()


# Third Party
import carb
from motion_planning_trajectory_execution.utils.helper import VoxelManager, add_extensions, add_robot_to_scene
from omni.isaac.core import World

from nvblox_torch.mapper import Mapper

# DEPRECATED:
# from omni.isaac.core.materials import OmniPBR
# INSTEAD:
from isaacsim.core.api.materials import OmniPBR  

from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.usd_helper import UsdHelper
from curobo.util.logger import setup_curobo_logger


# LOAD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", "camera_recorded_data3.pt")
LOAD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", "camera_data_4.pt")

# Load tensor file
frames = torch.load(LOAD_PATH, weights_only=False)
print(f"Loaded {len(frames)} frames")
print("Available keys:", frames[0].keys())

frame_idx = 0 # frame number from .pt file

if __name__ == "__main__":
    act_distance = 0.4
    voxel_size = 0.02
    render_voxel_size = 0.02
    clipping_distance = 0.5

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    my_world.scene.add_default_ground_plane(z_position=-1.0)

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    # stage.DefinePrim("/curobo", "Xform")

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    # target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    # target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))

    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "tsdf",
                    "voxel_size": 0.02,
                    # "min_bound": [-5.0, -5.0, -1.0],
                    # "max_bound": [5.0, 5.0, 5.0]
                }
            }
        }
    )

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, _ = add_robot_to_scene(robot_cfg, my_world) #, position=np.array([0, 0, 0.5]))

    articulation_controller = robot.get_articulation_controller()
    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg_table.cuboid[0].pose[2] -= 0.04
    # world_cfg.add_obstacle(world_cfg_table.cuboid[0])
    
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.03,
        collision_activation_distance=0.05,
        # acceleration_scale=1.0,
        # self_collision_check=True,
        # maximum_trajectory_dt=0.25,
        # finetune_dt_scale=1.05,
        # fixed_iters_trajopt=True,
        # finetune_trajopt_iters=300,
        # minimize_jerk=True,
    )

    motion_gen = MotionGen(motion_gen_config)
    print("warming up...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)
    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
    )

    world_model = motion_gen.world_collision

    # world_model = RobotWorldConfig.load_from_config(
    #     "franka.yml",
    #     world_cfg,
    #     collision_activation_distance=act_distance,
    #     collision_checker_type=collision_checker_type,
    # )

    # model = RobotWorld(world_model)

    camera_pose = Pose.from_list([0, 0, 0, 0, 0, 0, 0])

    cmd_plan = None
    cmd_idx = 0
    i = 0 # Simulation loop counter
    mapping_done = False

    if not args.use_debug_draw:
        voxel_viewer = VoxelManager(5000, size=render_voxel_size)

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if step_index % 5 == 0.0 and mapping_done == False:
            # world_model.decay_layer("world")
            if frame_idx < len(frames):
                current_frame = frames[frame_idx]

                # depth = tensor_args.to_device(current_frame["depth"])
                # intrinsics = tensor_args.to_device(current_frame["intrinsics"])
                # position = tensor_args.to_device(current_frame["position"])
                # quaternion = tensor_args.to_device(current_frame["quaternion"])
                
                print("position: ", current_frame["position_tensor"].cpu().numpy())
                print("quaternion: ", current_frame["quaternion_tensor"].cpu().numpy())

                # quaternion_corrected = np.array([current_frame["quaternion"][3],current_frame["quaternion"][0], current_frame["quaternion"][1], current_frame["quaternion"][2]])

                # new_position = current_frame["position_tensor"].cpu().numpy()
                # new_position[2] += 0.5

                camera_pose = Pose(
                    position=tensor_args.to_device(current_frame["position_tensor"].cpu().numpy()),
                    quaternion=tensor_args.to_device(current_frame["quaternion_tensor"].cpu().numpy()),
                )

                # camera_pose = Pose(
                #     position=tensor_args.to_device(new_position),
                #     quaternion=tensor_args.to_device(current_frame["quaternion_tensor"].cpu().numpy()),
                # )

                # camera_pose = Pose(
                #     position=tensor_args.to_device(current_frame["position"].cpu().numpy()),
                #     quaternion=tensor_args.to_device(quaternion_corrected),
                # )

                data_camera = CameraObservation(
                    rgb_image=current_frame["rgba_tensor"],
                    depth_image=current_frame["depth_tensor"],
                    intrinsics=current_frame["intrinsics_tensor"],
                    pose=camera_pose
                )
                # data_camera = CameraObservation(
                #     depth_image=depth,
                #     intrinsics=intrinsics,
                #     pose=camera_pose
                # )

                data_camera = data_camera.to(device=tensor_args.device)

                print(f"got frame {frame_idx+1}/{len(frames)}")

                frame_idx += 1

                world_model.add_camera_frame(data_camera, "world")
                world_model.process_camera_frames("world", False)
                # world_model.add_camera_frame(data_camera, "base_link")
                # world_model.process_camera_frames("base_link", False)
                torch.cuda.synchronize()
                world_model.update_blox_hashes()
                
                bounding = Cuboid("t", dims=[5.0, 5.0, 5.0], pose=[0, 0, 0, 1, 0, 0, 0])
                # Bounding box should a cube of at least [length_of_manip + clipping_distance]^3
                
                voxels = world_model.get_voxels_in_bounding_box(bounding, voxel_size)
                if voxels.shape[0] > 0:
                    # print("IM INSIDEEEEEEEEE")
                    # print(f"[DEBUG] Total voxels in world model: {voxels.shape[0]}")
                    # voxels = voxels[voxels[:, 2] > voxel_size]
                    voxels = voxels[voxels[:, 2] > -0.5]
                    # voxels = voxels[voxels[:, 2] > -0.1]
                    voxels = voxels[voxels[:, 0] > 0.0]
                    # voxels = voxels[voxels[:, 0] > 0.03 | voxels[:, 0] < -0.03]
                    # voxels = voxels[(voxels[:, 0] > 0.03) | (voxels[:, 0] < -0.03)]
                    # voxels = voxels[voxels[:, 1] > 0.03 | voxels[:, 1] < -0.03]
                    # voxels = voxels[(voxels[:, 1] > 0.03) | (voxels[:, 1] < -0.03)]

                    # voxels = voxels[voxels[:, 0] > 0.0]
                    # voxel_viewer.update_voxels(voxels[:, :3].cpu().numpy())
                    # Count after filtering
                    # print(f"[DEBUG] Voxels after filtering: {voxels.shape[0]}")
                    # voxels = voxels

                    if args.use_debug_draw:
                    #     draw_points(voxels)
                    # CHECK LATER, CLEAN IT!!!
                        pass
                    else:
                        # print("ALSO HEEEREEEEEE")
                        voxels = voxels.cpu().numpy()
                        voxel_viewer.update_voxels(voxels[:, :3])

                    voxel_viewer.update_voxels(voxels[:, :3])
                else:
                    if not args.use_debug_draw:
                        pass

            else:
                print("All frames processed")
                mapping_done = True
                
                # Save map as .nvblox
                # world_model.save_layer("world", "nvblox_map_test.nvblx")
                
                # # Save map as mesh
                # mesh = world_model.get_mesh_from_blox_layer("world", mode="nvblox")
                # print("Num vertices:", len(mesh.vertices))
                # print("Num faces:", len(mesh.faces))
                # print("Has colors:", mesh.vertex_colors is not None and len(mesh.vertex_colors) > 0)

                # # Convert to numpy
                # vertices = np.array(mesh.vertices, dtype=np.float32)
                # faces = np.array(mesh.faces, dtype=np.int32)

                # # If colors exist, convert too
                # colors = None
                # if hasattr(mesh, "vertex_colors") and mesh.vertex_colors is not None:
                #     colors = np.array(mesh.vertex_colors, dtype=np.uint8)

                # # Build trimesh object
                # trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)

                # # Export
                # trimesh_mesh.export("mesh_map_test2.obj")
                # print(f"✅ Mesh saved")


                # Mapper.save_map()

        
        if mapping_done == True:
            cube_position, cube_orientation = target.get_world_pose()

            if past_pose is None:
                past_pose = cube_position
            if target_pose is None:
                target_pose = cube_position
            sim_js = robot.get_joints_state()
            if sim_js is None:
                print("sim_js is None")
                continue
            sim_js_names = robot.dof_names

            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
            
            if (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                and np.linalg.norm(past_pose - cube_position) == 0.0
                and np.max(np.abs(sim_js.velocities)) < 0.5 ## BEFORE IT WAS 0.2 WHICH IS TOO LOW
            ):
                # Set EE teleop goals, use cube for simple non-vr init:
                ee_translation_goal = cube_position
                ee_orientation_teleop_goal = cube_orientation

                # compute curobo solution:
                ik_goal = Pose(
                    position=tensor_args.to_device(ee_translation_goal),
                    quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
                )

                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

                succ = result.success.item()  # ik_result.success.item()

                if succ:
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = motion_gen.get_full_js(cmd_plan)
                    # get only joint names that are in both:
                    idx_list = []
                    common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(robot.get_dof_index(x))
                            common_js_names.append(x)
                    # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                    cmd_idx = 0

                else:
                    carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
                target_pose = cube_position
            
            past_pose = cube_position

            if cmd_plan is not None:
                cmd_state = cmd_plan[cmd_idx]

                # get full dof state
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )
                # set desired joint angles obtained from IK:
                articulation_controller.apply_action(art_action)
                cmd_idx += 1

                for _ in range(2):
                    my_world.step(render=False)
                
                if cmd_idx >= len(cmd_plan.position):
                    cmd_idx = 0
                    cmd_plan = None
                    
        
    print("finished program")

    simulation_app.close()



# Inspect all frames with OpenCV
# for i, frame in enumerate(frames):
#     depth = frame["depth"].cpu().numpy()             # [480, 640]
#     intrinsics = frame["intrinsics"].cpu().numpy()   # [3, 3]
#     pose = frame["pose"]

#     # Simple depth colormap visualization
#     depth_colormap = cv2.applyColorMap(
#         cv2.convertScaleAbs(depth, alpha=100), cv2.COLORMAP_VIRIDIS
#     )

#     # Display depth
#     cv2.imshow("Depth", depth_colormap)
#     print(f"Frame {i}: Intrinsics:\n{intrinsics}\nPose: {pose}")

#     key = cv2.waitKey(100)  # 100 ms per frame
#     if key & 0xFF == ord("q") or key == 27:  # q or ESC to quit early
#         break

# cv2.destroyAllWindows()

