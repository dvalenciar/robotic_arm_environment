'''

Author: David Valencia
Date: 19/Sep/2021


Describer:  
			This is the main launch file for my environment simulation. 
			
			
			Here start the simulation and call the previously developed launch of both sphere and 
			robot arm and spawn them together in a GAZEBO. 
			
			my_doosan_pkg --> my_doosan_controller.launch.py
			my_sphere_pkg --> my_sphere.launch.py
				
			Just call the launch files, for details of config please go to the sphere pkg or Doosan pkg 

			To avoid errors, I start rviz in another terminal manually  
'''

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess



def generate_launch_description():

	my_sphere_files       = get_package_share_directory('my_sphere_pkg')
	my_doosan_robot_files = get_package_share_directory('my_doosan_pkg')
	my_environmets_files  = get_package_share_directory('my_environment_pkg')

	# Start doosan robot and controller
	doosan_robot = IncludeLaunchDescription(PythonLaunchDescriptionSource(my_doosan_robot_files + '/launch/my_doosan_controller.launch.py')) 
	
	# Start sphere mark
	sphere_mark  = IncludeLaunchDescription(PythonLaunchDescriptionSource(my_sphere_files + '/launch/my_sphere.launch.py')) 


	'''
	# Start Rviz
	rviz_file = my_environmets_files + "/rviz/my_rviz_env.rviz"
	rviz_node = Node( package='rviz2',
					  executable='rviz2',
					  name='rviz2',
					  output='log',
					  arguments=['-d', rviz_file])
	'''

	# Start Gazebo   
	world_file_name = 'my_world.world'
	world = os.path.join(get_package_share_directory('my_environment_pkg'), 'worlds', world_file_name)
	gazebo_node = ExecuteProcess(cmd=['gazebo', '--verbose', world,'-s', 'libgazebo_ros_factory.so'], output='screen')


	# Node 


	ld = LaunchDescription()

	ld.add_action (doosan_robot)
	ld.add_action (sphere_mark)
	ld.add_action (gazebo_node)
	#ld.add_action (rviz_node)

	return ld
	
	