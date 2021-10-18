'''
Author: David Valencia
Date: 25 / 08 /2021

Describer:  Simple launch to SIMULATE the doosan robot in GAZEBO in my own package

			Based on the original git package from doosan-robot2 

			This scripts just spawns the robot arm in GAZEBO 

			the robot description (urdf and xacro) are in:
			
			src/my_doosan_pkg/description/xacro
			
			Robot model m1013 color white
			Robot model a0912 color blue	
'''

import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import Command
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

	#robot model to option m1013 or a0912 
	
	robot_model = 'a0912'
	#robot_model = 'm1013'

	xacro_file = get_package_share_directory('my_doosan_pkg') + '/description'+'/xacro/'+ robot_model +'.urdf.xacro'

	
	# Robot State Publisher 
	robot_state_publisher = Node(package    ='robot_state_publisher',
								 executable ='robot_state_publisher',
								 name       ='robot_state_publisher',
								 output     ='both',
								 parameters =[{'robot_description': Command(['xacro', ' ', xacro_file])           
								}])


	# Spawn the robot in Gazebo
	spawn_entity_robot = Node(package    ='gazebo_ros', 
							  executable ='spawn_entity.py', 
							  arguments  = ['-entity', 'my_doosan_robot', '-topic', 'robot_description'],
							  output     ='screen')

	# Start Gazebo with my empty world   
	world_file_name = 'my_empty_world.world'
	world = os.path.join(get_package_share_directory('my_doosan_pkg'), 'worlds', world_file_name)
	gazebo_node = ExecuteProcess(cmd=['gazebo', '--verbose', world,'-s', 'libgazebo_ros_factory.so'], output='screen')
	


	return LaunchDescription([robot_state_publisher, spawn_entity_robot, gazebo_node ])