'''
Author: David Valencia
Date: 15 / 10 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:
            Spawns the robot in Gazebo (gz-sim / Harmonic) and loads + starts a
            joint_state_broadcaster and a joint_trajectory_controller.

            The controller configuration lives in:
                my_doosan_pkg/config/simple_controller.yaml

            Just for testing purposes inside this package only.

            - Robot model m1013 color white
            - Robot model a0912 color blue
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    robot_model = 'a0912'
    # robot_model = 'm1013'

    pkg_share = get_package_share_directory('my_doosan_pkg')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_share, 'description', 'xacro', robot_model + '.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'my_empty_world.world')

    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro', ' ', xacro_file]), value_type=str
        )
    }

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
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

    # Bridge the simulation clock so ROS nodes use sim time
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
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
        gazebo,
        robot_state_publisher,
        clock_bridge,
        spawn_entity_robot,
        spawn_then_jsb,
        jsb_then_jtc,
    ])