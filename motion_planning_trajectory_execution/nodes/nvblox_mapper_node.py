import rclpy
from motion_planning_trajectory_execution.ros.nvblox_mapper import *

def main():
    rclpy.init()
    node = NvbloxMapper()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.step()
    except KeyboardInterrupt:
        pass
    
    # After exitng Ctrl+C
    node.save_all()
    node.realsense.stop_device()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
