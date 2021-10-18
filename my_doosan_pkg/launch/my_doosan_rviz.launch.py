'''
Author: David Valencia
Date: 25 / 08 /2021

Describer:  Simple launch to visulizate the doosan robot in RVIZ in my own package
		
			This file open Rviz, vizualizate the robot and open a GUI intergace to move the joints

			- Robot model m1013 color white
			- Robot model a0912 color blue
			
			Based on the original git package from doosan-robot2 
			
'''

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

	#robot model to option m1013 or a0912 
	
	robot_model = 'a0912'
	#robot_model = 'm1013'

	xacro_file = get_package_share_directory('my_doosan_pkg') + '/description'+'/xacro/'+ robot_model +'.urdf.xacro'


	# RViz
	rviz_config_file = get_package_share_directory('my_doosan_pkg') + "/rviz/view_config.rviz"
	rviz_node = Node(package    ='rviz2',
					 executable ='rviz2',
					 name       ='rviz2',
					 output     ='log',
					 arguments  =['-d', rviz_config_file])


	'''
	# Robot State Publisher 
	robot_state_publisher = Node(package='robot_state_publisher',
								 executable='robot_state_publisher',
								 name='robot_state_publisher',
								 output='screen',
								 parameters=[{'robot_description': robot_desc
								}])
	'''

	# Robot State Publisher 
	robot_state_publisher = Node(package    ='robot_state_publisher',
								 executable ='robot_state_publisher',
								 name       ='robot_state_publisher',
								 output     ='both',
								 parameters =[{'robot_description': Command(['xacro', ' ', xacro_file])           
								}])


	# Joint State Publisher 
	joint_state_publisher_gui = Node(package  ='joint_state_publisher_gui',
									executable='joint_state_publisher_gui',
									output    ='screen',
									name      ='joint_state_publisher_gui')


	return LaunchDescription([robot_state_publisher, joint_state_publisher_gui, rviz_node ])