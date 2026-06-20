'''
Author: David Valencia
Date: 2026-06 (ROS 2 Jazzy + Gazebo Harmonic)

Describer:  Standalone test launch for the goal sphere package.

            Unlike my_sphere.launch.py (a building block that assumes gz-sim is
            already running and is meant to be included by the RL environment),
            this launch is self-contained: it starts gz-sim with an empty world,
            then includes my_sphere.launch.py to spawn the sphere, start the
            ros_gz bridge, and run the marker node.

            Use it to test the sphere package on its own:
                ros2 launch my_sphere_pkg my_sphere_standalone.launch.py
            then teleport the sphere with:
                ros2 run my_sphere_pkg my_client_node
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('my_sphere_pkg')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(pkg_dir, 'worlds', 'my_empty_world.world')

    # Gazebo (gz-sim / Harmonic)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -v 4 ', world_file]}.items(),
    )

    # Bridge the simulation clock so ROS nodes use sim time
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # Reuse the building block: spawn sphere + ros_gz bridge + marker node
    sphere = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'my_sphere.launch.py')
        )
    )

    return LaunchDescription([gazebo, clock_bridge, sphere])
