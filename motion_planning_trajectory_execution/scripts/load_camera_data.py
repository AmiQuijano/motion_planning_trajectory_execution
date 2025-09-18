#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to load .pt Tensor file ccoming from camera_recorder.py and inspect its content
# Run with omni_python

import torch
import os
import cv2

from curobo.types.math import Pose


LOAD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "camera_data", "camera_recorded_data3.pt")

# Load tensor file
# torch.serialization.add_safe_globals([Pose])
frames = torch.load(LOAD_PATH, weights_only=True)
print(f"Loaded {len(frames)} frames")
print("Available keys:", frames[0].keys())

# Inspect all frames with OpenCV
for i, frame in enumerate(frames):
    depth = frame["depth"].cpu().numpy()             # [480, 640]
    intrinsics = frame["intrinsics"].cpu().numpy()   # [3, 3]
    position = frame["position"].cpu().numpy() 
    quaternion = frame["quaternion"].cpu().numpy() 

    # Simple depth colormap visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=100), cv2.COLORMAP_VIRIDIS
    )

    print(position)
    print(quaternion)
    # Display depth
    cv2.imshow("Depth", depth_colormap)
    # print(f"Frame {i}: Intrinsics:\n{intrinsics}\nPose: {pose}")

    key = cv2.waitKey(100)  # 100 ms per frame
    if key & 0xFF == ord("q") or key == 27:  # q or ESC to quit early
        break

cv2.destroyAllWindows()