# Script to verify that RealsenseDataloader launches correctly

from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

rs_loader = RealsenseDataloader(clipping_distance_m=0.8)

for i in range(10):
    try:
        data = rs_loader.get_data()
        print("Got frame:", i, data["depth"].shape, data["intrinsics"].shape)
    except StopIteration:
        print("No frame yet")
