'''
Author: David Valencia
Date: 19/Sep/2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:
            This is the main launch file for the environment simulation.

            It starts gz-sim with an empty world and a /clock bridge, then
            includes the previously migrated building-block launches that spawn
            the robot arm (+ controllers) and the goal sphere (+ bridge + marker)
            into that running world:

                my_doosan_pkg --> my_doosan_controller.launch.py
                my_sphere_pkg --> my_sphere.launch.py

            For details of each piece see the sphere pkg or Doosan pkg.
            (RViz is started manually in another terminal.)
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    my_sphere_files = get_package_share_directory('my_sphere_pkg')
    my_doosan_robot_files = get_package_share_directory('my_doosan_pkg')
    my_environment_files = get_package_share_directory('my_environment_pkg')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(my_environment_files, 'worlds', 'my_world.world')

    # Start Gazebo (gz-sim / Harmonic)
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

    # Spawn the doosan robot + start its controllers (assumes gz running)
    doosan_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_doosan_robot_files, 'launch', 'my_doosan_controller.launch.py')
        )
    )

    # Spawn the goal sphere + ros_gz bridge + marker node (assumes gz running)
    sphere_mark = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_sphere_files, 'launch', 'my_sphere.launch.py')
        )
    )

    ld = LaunchDescription()
    ld.add_action(gazebo)
    ld.add_action(clock_bridge)
    ld.add_action(doosan_robot)
    ld.add_action(sphere_mark)

    return ld