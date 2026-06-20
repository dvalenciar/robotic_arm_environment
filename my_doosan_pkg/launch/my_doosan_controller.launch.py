'''
Author: David Valencia
Date: 26 / 08 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:
            This script spawns the robot and LOADS + STARTS a basic
            joint_state_broadcaster and joint_trajectory_controller.
            The controller configuration lives in:
                my_doosan_pkg/config/simple_controller.yaml

            Note: Gazebo and RViz are intentionally NOT started here.
                  This launch file is meant to be INCLUDED from the RL
                  environment launch file, which starts gz-sim + the world
                  itself. It only:
                    - publishes the robot_description
                    - spawns the robot into the already-running gz-sim
                    - brings up the controllers (controller_manager lives
                      inside the gz_ros2_control plugin, so the spawners must
                      run AFTER the robot is created).

            - Robot model m1013 color white
            - Robot model a0912 color blue
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    robot_model = 'a0912'
    # robot_model = 'm1013'

    pkg_share = get_package_share_directory('my_doosan_pkg')
    xacro_file = os.path.join(pkg_share, 'description', 'xacro', robot_model + '.urdf.xacro')

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

    # Spawn the robot from the /robot_description topic into the running gz-sim
    spawn_entity_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'my_doosan_robot'],
        output='screen',
    )

    # Controller spawners (controller_manager runs inside the gz_ros2_control plugin)
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    load_joint_trajectory_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Sequence: spawn entity -> joint_state_broadcaster -> joint_trajectory_controller
    spawn_then_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity_robot,
            on_exit=[load_joint_state_broadcaster],
        )
    )

    jsb_then_jtc = RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_joint_trajectory_controller],
        )
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_entity_robot,
        spawn_then_jsb,
        jsb_then_jtc,
    ])
