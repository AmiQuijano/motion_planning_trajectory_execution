from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_name = 'motion_planning_trajectory_execution'
    urdf_file = "hero_v2_alone.urdf" #'rm75_6f_realsense.urdf' #'rm75_6f.urdf' # 'hero_arm_v2.urdf'
    # urdf_path = os.path.join(get_package_share_directory(pkg_name), 'robot_description', 'hero_v3_3links', 'urdf', urdf_file)
    # urdf_path = os.path.join(get_package_share_directory(pkg_name), 'robot_description', urdf_file)
    urdf_path = "/home/ami/ros2_ws/src/motion_planning_trajectory_execution/robot_description/hero_v3/urdf/hero_7dof_v3_SW.urdf"
    # urdf_path = "/home/ami/ros2_ws/src/motion_planning_trajectory_execution/robot_description/hero_v2_alone.urdf"

    # Read the URDF content
    with open(urdf_path, 'r') as inf:
        robot_desc = inf.read()

    return LaunchDescription([
        # Use joint_state_publisher_gui for visual joint control
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        ),

        # Publish robot description to robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],  # Pass URDF here too
        ),
        
        # Launch RViz2 for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
