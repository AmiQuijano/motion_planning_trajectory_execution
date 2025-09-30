"""Launch realsense2_camera node."""
from launch import LaunchDescription
from  launch_ros.actions import Node
import os
import xacro
import tempfile
from ament_index_python.packages import get_package_share_directory

def to_urdf(xacro_path, parameters=None):
    """Convert the given xacro file to URDF file.
    * xacro_path -- the path to the xacro file
    * parameters -- to be used when xacro file is parsed.
    """
    urdf_path = tempfile.mktemp(prefix="%s_" % os.path.basename(xacro_path))

    # open and process file
    doc = xacro.process_file(xacro_path)
    # open the output file
    out = xacro.open_output(urdf_path)
    out.write(doc.toprettyxml(indent='  '))

    return urdf_path

def generate_launch_description():
    xacro_path = os.path.join(get_package_share_directory('realsense2_description'), 'urdf', 'test_d435i_camera.urdf.xacro')
    # xacro_path = os.path.join(get_package_share_directory('tesseract_moonshot'), 'urdf', 'rm75_6f_realsense.xacro')
    # xacro_path = os.path.join(get_package_share_directory('tesseract_moonshot'), 'urdf', 'hero_arm_v2.xacro')
    # xacro_path = os.path.join(get_package_share_directory('realman_interface'), 'urdf', 'realman_75.xacro')
    # xacro_path = os.path.join(get_package_share_directory('realsense2_description'), 'urdf', 'test_d455_camera.urdf.xacro')

    urdf = to_urdf(xacro_path)

    return LaunchDescription(
        [
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            arguments=[urdf]
            # parameters=[{'robot_description': urdf}],
        ),

        Node(
            name='model_node',
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='',
            output='screen',
            arguments=[urdf]
        )
    ])