'''

Author: David Valencia
Date: 07/ 04 /2022

Modification: 07/ 04 /2022

Describer: 

		Main environment v2.0
		
		This script is the main environment of my project. Here is the body of the code
		state state, action samples, data from sensor, reset request all are generated/read here.

		I use an action client to move the robot arm while using a client service to move the target point (green sphere).

		Also, the state space is created with the end-effector position, the sphere (target) position and the joint state.
		if the robot reaches the goal or the number of steps in each episode finishes the environment will reset i.e robot
		will move to target position and the target point will move to a new random location.

		To summarize, this script does:

			1) Return the state space
			2) Return the reward (distance between target and end-effector)
			3) Generate and return random action
			4) Reset the environment (robot to home position and target to new position

		However in order to be me organized I just create the class here without node and to run
		

		To run this environmet --> I create a second script called: run_environmet.py or collection_data.py  

		Executable name in the setup file: no needed here cause i will run the second script
'''

import os
import sys
import time
import rclpy
import random
import numpy as np
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState

import tf2_ros 
from tf2_ros import TransformException

from rclpy.action        import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from rclpy.duration import Duration




class MyRLEnvironmentNode(Node):

	def __init__ (self):

		super().__init__('node_main_rl_environment')

		print ("initializing.....")
		
		# end-effector transformation
		self.tf_buffer   = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


		# --------------------------Client for reset the sphere position --------------------------#
		self.client_reset_sphere = self.create_client(SetEntityState,'/gazebo/set_entity_state')
		while not self.client_reset_sphere.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('sphere reset-service not available, waiting...')
		self.request_sphere_reset = SetEntityState.Request()


		# ------------------------- Action-client to change joints position -----------------------#
		self.trajectory_action_client = ActionClient (self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')


		# --------------------------Subcribers topics --------------------------------------------#

		# Subcribe topic with the joints states
		self.joint_state_subscription = message_filters.Subscriber(self, JointState, '/joint_states')
		
		# Subcribe topic with the sphere position
		self.target_point_subscription = message_filters.Subscriber(self, ModelStates, '/gazebo/model_states')

		# Create the message filter (if a msg is detected for each subcriber, do the callback)
		#self.ts = message_filters.TimeSynchronizer([self.joint_state_subscription, self.target_point_subscription], queue_size=30)
		self.ts = message_filters.ApproximateTimeSynchronizer([self.joint_state_subscription, self.target_point_subscription], queue_size=10, slop=0.1, allow_headerless=True)
		self.ts.registerCallback(self.initial_callback)



	def initial_callback(self, joint_state_msg, target_point_msg):

		# Seems that the order that the joint values arrive is: ['joint2', 'joint3', 'joint1', 'joint4', 'joint5', 'joint6']
		
		# Position of each joint:
		self.joint_1_pos = joint_state_msg.position[2]
		self.joint_2_pos = joint_state_msg.position[0]
		self.joint_3_pos = joint_state_msg.position[1]
		self.joint_4_pos = joint_state_msg.position[3]
		self.joint_5_pos = joint_state_msg.position[4]
		self.joint_6_pos = joint_state_msg.position[5]

		# Velocity of each joint:
		self.joint_1_vel =  joint_state_msg.velocity[2]
		self.joint_2_vel =  joint_state_msg.velocity[0]
		self.joint_3_vel =  joint_state_msg.velocity[1]
		self.joint_4_vel =  joint_state_msg.velocity[3]
		self.joint_5_vel =  joint_state_msg.velocity[4]
		self.joint_6_vel =  joint_state_msg.velocity[5]

		# Determine the sphere position in Gazebo wrt world frame
		sphere_index = target_point_msg.name.index('my_sphere') # Get the corret index for the sphere
		self.pos_sphere_x = target_point_msg.pose[sphere_index].position.x 
		self.pos_sphere_y = target_point_msg.pose[sphere_index].position.y 
		self.pos_sphere_z = target_point_msg.pose[sphere_index].position.z 

		# Determine the pose(position and location) of the end-effector w.r.t. world frame
		self.robot_x, self.robot_y, self.robot_z = self.get_end_effector_transformation()



	def get_end_effector_transformation(self):

		# Determine the pose(position and location) of the end effector w.r.t. world frame

		try:
			now = rclpy.time.Time()	
			self.reference_frame = 'world'
			self.child_frame     = 'link6'
			trans = self.tf_buffer.lookup_transform(self.reference_frame, self.child_frame, now) # This calculate the position of the link6 w.r.t. world frame
			
		except TransformException as ex:
			self.get_logger().info(f'Could not transform {self.reference_frame} to {self.child_frame}: {ex}')
			return

		else:
			# Traslation 
			ef_robot_x = trans.transform.translation.x
			ef_robot_y = trans.transform.translation.y
			ef_robot_z = trans.transform.translation.z
			#print ('Translation: [',round (ef_robot_x,3), round(ef_robot_x,3), round(ef_robot_x,3),']')
			
			# Rotation
			ef_qx = trans.transform.rotation.x
			ef_qy = trans.transform.rotation.y
			ef_qz = trans.transform.rotation.z
			ef_qw = trans.transform.rotation.w
			#print ('Rotation: in Quaternion [',round (ef_qx,3), round(ef_qy,3), round(ef_qz,3), round(ef_qw,3),']')

			return round (ef_robot_x,3), round(ef_robot_y,3), round(ef_robot_z,3)

	
	def reset_environment_request(self):

		# Every time this function is called a request to the Reset the Environment is sent 
		# i.e. Move the robot to home position and change the
		# sphere location and waits until get response/confirmation

		# -------------------- reset sphere position------------------#

		# For now the sphere's position will be inside a 1x1x1 workspace in front of the robot 
		sphere_position_x = random.uniform( 0.05, 1.05)
		sphere_position_y = random.uniform( -0.5, 0.5)
		sphere_position_z = random.uniform( 0.05, 1.05)

		self.request_sphere_reset.state.name = 'my_sphere'
		self.request_sphere_reset.state.reference_frame = 'world'
		self.request_sphere_reset.state.pose.position.x = sphere_position_x
		self.request_sphere_reset.state.pose.position.y = sphere_position_y
		self.request_sphere_reset.state.pose.position.z = sphere_position_z
		
		self.future_sphere_reset = self.client_reset_sphere.call_async(self.request_sphere_reset)

		self.get_logger().info('Reseting sphere to new position...')

		rclpy.spin_until_future_complete(self, self.future_sphere_reset)

		sphere_service_response = self.future_sphere_reset.result()
		
		if sphere_service_response.success:
			self.get_logger().info("Sphere Moved to a New Possiton Success")
		else:
			self.get_logger().info("Sphere Reset Request failed")

		
		#---------------------reset robot position-------------------#

		home_point_msg = JointTrajectoryPoint()
		home_point_msg.positions     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.velocities    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.time_from_start = Duration(seconds=2).to_msg()

		joint_names   = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
		home_goal_msg = FollowJointTrajectory.Goal()
		home_goal_msg.goal_time_tolerance    = Duration(seconds=1).to_msg()
		home_goal_msg.trajectory.joint_names = joint_names
		home_goal_msg.trajectory.points      = [home_point_msg]
		
		self.trajectory_action_client.wait_for_server()  # waits for the action server to be available
		
		send_home_goal_future = self.trajectory_action_client.send_goal_async(home_goal_msg) # Sending home-position request
		
		rclpy.spin_until_future_complete(self, send_home_goal_future) # Wait for goal status
		goal_reset_handle = send_home_goal_future.result()

		if not goal_reset_handle.accepted:
			self.get_logger().info(' Home-Goal rejected ')
			return
		self.get_logger().info('Moving robot to home position...')

		get_reset_result = goal_reset_handle.get_result_async()
		rclpy.spin_until_future_complete(self, get_reset_result)  # Wait for response

		if get_reset_result.result().result.error_code == 0:
			self.get_logger().info('Robot in Home position without problems')
		else:
			self.get_logger().info('There was a problem with the action')


	def action_step_service(self, action_values):
		
		# Every time this function is called, it passes the action vector (desire position of each joint) 
		# to the action-client to execute the trajectory
		
		points = []

		point_msg = JointTrajectoryPoint()
		point_msg.positions     = action_values
		point_msg.velocities    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		point_msg.time_from_start = Duration(seconds=2.0).to_msg() # be careful about this time 
		points.append(point_msg) 

		joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
		goal_msg    = FollowJointTrajectory.Goal()
		goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg() # goal_time_tolerance allows some freedom in time, so that the trajectory goal can still
															        # succeed even if the joints reach the goal some time after the precise end time of the trajectory.
															
		goal_msg.trajectory.joint_names = joint_names
		goal_msg.trajectory.points      = points

		self.get_logger().info('Waiting for action server to move the robot...')
		self.trajectory_action_client.wait_for_server() # waits for the action server to be available

		self.get_logger().info('Sending goal-action request...')
		self.send_goal_future = self.trajectory_action_client.send_goal_async(goal_msg) 

		self.get_logger().info('Checking if the goal is accepted...')
		rclpy.spin_until_future_complete(self, self.send_goal_future ) # Wait for goal status

		goal_handle = self.send_goal_future.result()

		if not goal_handle.accepted:
			self.get_logger().info(' Action-Goal rejected ')
			return
		self.get_logger().info('Action-Goal accepted')

		self.get_logger().info('Checking the response from action-service...')
		self.get_result = goal_handle.get_result_async()
		rclpy.spin_until_future_complete(self, self.get_result ) # Wait for response

		if self.get_result.result().result.error_code == 0:
			self.get_logger().info('Action Completed without problem')
		else:
			self.get_logger().info('There was a problem with the accion')


	def generate_action_funct(self):

		# This is a continuous action space
		# This function generates random values in radians for each joint
		# These values range were tested in advance to make sure that there were not internal collisions 

		angle_j_1 = random.uniform( -3.14159, 3.14159)   # values in degrees [ -180, 180]
		angle_j_2 = random.uniform( -0.57595, 0.57595)	 # values in degrees [  -33,  33]
		angle_j_3 = random.uniform( -2.51327, 2.51327)   # values in degrees [ -144, 144]
		angle_j_4 = random.uniform( -3.14159, 3.14159)   # values in degrees [ -180, 180]
		angle_j_5 = random.uniform( -3.14159, 3.14159)   # values in degrees [ -180, 180]
		angle_j_6 = random.uniform( -3.14159, 3.14159)   # values in degrees [ -180, 180]

		return [angle_j_1, angle_j_2, angle_j_3, angle_j_4, angle_j_5, angle_j_6]


	def calculate_reward_funct(self):

		# I aim with this function to get the reward value. For now, the reward is based on the distance
		# i.e. Calculate the euclidean distance between the link6 (end effector) and sphere (target point)
		# and each timestep the robot receives -1 but if it reaches the goal (distance < 0.05) receives +10


		try:
			robot_end_position    = np.array((self.robot_x, self.robot_y, self.robot_z))
			target_point_position = np.array((self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z))
			
		except: 
			self.get_logger().info('could not calculate the distance yet, trying again...')
			return

		else:
			distance = np.linalg.norm(robot_end_position - target_point_position)
			
			if distance <= 0.05:
				self.get_logger().info('Goal Reached')
				done = True
				reward_d = 10
			else:
				done = False
				reward_d = -1

			return reward_d, done


	def state_space_funct (self):

		# This function creates the state state vector and returns the current value of each variable
		# i.e end-effector position, each joint value, target (sphere position) 

		try:
			state = [
					self.robot_x, self.robot_y, self.robot_z, 
					self.joint_1_pos, self.joint_2_pos, self.joint_3_pos, self.joint_4_pos, self.joint_5_pos, self.joint_6_pos, 
					self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z
					]			 	
		except:
			
			self.get_logger().info('-------node not ready yet, Still getting values------------------')
			return 
		else:
			return state

