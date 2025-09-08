import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'motion_planning_trajectory_execution'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + "/nodes", package_name + "/ros"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all files that may be used in scripts through get_package_share_directory
        (os.path.join('share', package_name, 'configs'), glob('configs/*', recursive=True)),
        (os.path.join('share', package_name, 'trajectories'), glob('trajectories/*', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ami Quijano Shimizu',
    maintainer_email='ami.quijanoshimizu@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'trajectory_execution_node = {package_name}.nodes.trajectory_execution_node:main', 
            f'test_ros2_curobo = {package_name}.nodes.test_ros2_curobo:main', 
        ],
    },
)