"""
Author: David Valencia
Date: 25 / 08 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:  Simple launch to SIMULATE the doosan robot in GAZEBO in my own package.
            Based on the original git package from doosan-robot2.
            This script just spawns the robot arm in GAZEBO (gz-sim / Harmonic),
            no controllers and no RViz.
            The robot description (urdf and xacro) are in: src/my_doosan_pkg/description/xacro

            Robot model m1013 color white.
            Robot model a0912 color blue.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # robot model to option m1013 or a0912

    robot_model = 'a0912'
    # robot_model = 'm1013'

    pkg_share = get_package_share_directory('my_doosan_pkg')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_share, 'description', 'xacro', robot_model + '.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'my_empty_world.world')

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro', ' ', xacro_file]), value_type=str),
            'use_sim_time': True,
        }],
    )

    # Gazebo (gz-sim / Harmonic), started through ros_gz_sim
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -v 4 ', world_file]}.items(),
    )

    # Spawn the robot from the /robot_description topic
    spawn_entity_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'my_doosan_robot'],
        output='screen',
    )

    return LaunchDescription([gazebo, robot_state_publisher, spawn_entity_robot])