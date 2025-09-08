import rclpy
from motion_planning_trajectory_execution.ros.trajectory_execution import *

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