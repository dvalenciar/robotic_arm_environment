'''
Author: David Valencia
Date: 11 / 08 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:  This script spawns the sphere in Gazebo (gz-sim / Harmonic) from an
            SDF file, starts the ros_gz_bridge for the sphere (pose read + set_pose
            service), and runs the marker node that republishes the sphere position
            for RViz.

            Migration note:
              - Spawning now uses ros_gz_sim/create instead of gazebo_ros/spawn_entity.py
              - Reading/setting the pose goes through ros_gz_bridge (config/sphere_bridge.yaml)

            Note: Gazebo and the empty world are NOT started here on purpose.
                  This launch file is meant to be INCLUDED from the RL environment
                  launch file, which starts gz-sim + the world itself.
'''

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('my_sphere_pkg')

    sdf_file = os.path.join(pkg_dir, 'models', 'sdf', 'sphere_goal', 'model.sdf')
    bridge_config = os.path.join(pkg_dir, 'config', 'sphere_bridge.yaml')

    # Spawn the sphere into the (already running) gz-sim world
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-file', sdf_file, '-name', 'my_sphere',
                   '-x', '0.5', '-y', '0.5', '-z', '1.0'],
        output='screen',
    )

    # Bridge: set_pose service + pose/info topic (replaces Classic gazebo_ros plugins)
    sphere_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
    )

    # node_mark -> coordinate_node.py -> reads the sphere pose and publishes the Marker topic
    node_mark = Node(package='my_sphere_pkg', executable='reader_mark_node', output='screen')

    return LaunchDescription([spawn_entity, sphere_bridge, node_mark])