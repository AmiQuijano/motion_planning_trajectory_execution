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

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

import numpy as np
import torch
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
# Standard Library
import argparse

# Third Party
import carb
from motion_planning_trajectory_execution.utils.helper import VoxelManager #, add_robot_to_scene
from omni.isaac.core import World

# DEPRECATED:
# from omni.isaac.core.materials import OmniPBR
# INSTEAD:
from isaacsim.core.api.materials import OmniPBR  

from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.usd_helper import UsdHelper

parser = argparse.ArgumentParser()

# parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--use-debug-draw",
    action="store_true",
    help="When True, sets robot in static mode",
    default=False,
)
args = parser.parse_args()

LOAD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", "camera_recorded_data3.pt")

# Load tensor file
frames = torch.load(LOAD_PATH, weights_only=True)
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
    my_world.scene.add_default_ground_plane()

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

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
                }
            }
        }
    )

    world_model = RobotWorldConfig.load_from_config(
        "franka.yml",
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=collision_checker_type,
    )

    model = RobotWorld(world_model)

    camera_pose = Pose.from_list([0, 0, 0, 0, 0, 0, 0])

    i = 0 # Simulation loop counter
    tensor_args = TensorDeviceType()

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

        if step_index % 5 == 0.0:
            model.world_model.decay_layer("world")
            if frame_idx < 3: #len(frames):
                current_frame = frames[frame_idx]

                # depth = tensor_args.to_device(current_frame["depth"])
                # intrinsics = tensor_args.to_device(current_frame["intrinsics"])
                # position = tensor_args.to_device(current_frame["position"])
                # quaternion = tensor_args.to_device(current_frame["quaternion"])
                                            
                camera_pose = Pose(
                    position=tensor_args.to_device(current_frame["position"].cpu().numpy()),
                    quaternion=tensor_args.to_device(current_frame["quaternion"].cpu().numpy()),
                )

                print("position: ", current_frame["position"].cpu().numpy())
                print("quaternion: ", current_frame["quaternion"].cpu().numpy())

                data_camera = CameraObservation(
                    depth_image=current_frame["depth"],
                    intrinsics=current_frame["intrinsics"],
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

                model.world_model.add_camera_frame(data_camera, "world")
                model.world_model.process_camera_frames("world", False)
                torch.cuda.synchronize()
                model.world_model.update_blox_hashes()
                
                bounding = Cuboid("t", dims=[3.0, 3.0, 3.0], pose=[0, 0, 0, 1, 0, 0, 0])
                # Bounding box should a cube of at least [length_of_manip + clipping_distance]^3
                
                voxels = model.world_model.get_voxels_in_bounding_box(bounding, voxel_size)
                if voxels.shape[0] > 0:
                    # print("IM INSIDEEEEEEEEE")
                    # print(f"[DEBUG] Total voxels in world model: {voxels.shape[0]}")
                    voxels = voxels[voxels[:, 2] > voxel_size]
                    voxels = voxels[voxels[:, 0] > 0.0]
                    # Count after filtering
                    # print(f"[DEBUG] Voxels after filtering: {voxels.shape[0]}")

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
                # print("All frames processed")
                pass
                

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

