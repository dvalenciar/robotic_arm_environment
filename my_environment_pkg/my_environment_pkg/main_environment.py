"""
Author: David Valencia
Date: 04 / 10 /2021
Describer:

		This script is the main environment of my project. Here is the body of the code
		but to run this I create a second file called run_environmet.py

		Here I call the necessary topics (publisher and subscribers).

		I use an action client to move the robot arm while using a client service to move the target point.

		Also, the state space is created with the end-effector position, the sphere (target) position and the joint state.
		if the robot reaches the goal or the number of steps in each episode finishes the environment will reset i.e robot
		will move to target position and the target point will move to a new random location.

		To summarize, this script does:

			1) Return the state space
			2) Return the reward (distance between target and end-effector)
			3) Generate and return random action
			4) Reset the environment (robot to home position and target to new position)
			5) A collition topic is preseted here in case the robot crash with floor

		update 5/04/2022 

		maybe there is a mistake in action_step_service  seems that i need to be very carefull with the duration time
		additionally call the on_timer_transformation function is not optimall, so i may be other way
		
		update 5/04/2022:
		
		Please see/use main_rl_environment.py instead of this script
"""

import time
import random
import numpy as np 

import rclpy
from rclpy.node import Node

import tf2_ros 
from tf2_ros import TransformException

from builtin_interfaces.msg import Duration

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ContactsState

from gazebo_msgs.srv import SetEntityState

from rclpy.action        import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class MyEnvironmentNode(Node):
	
	def __init__(self):

		super().__init__('node_main_environment')

		self.tf_buffer   = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		# Call on_timer_transformation function every 0.1 second
		self.timer = self.create_timer(0.1, self.on_timer_transformation)

		# Subcribe topic with the sphere position
		self.target_point_subscription = self.create_subscription(ModelStates, '/gazebo/model_states', self.target_state_callback, 1)

		# Subcribe topic with the joint states
		self.joint_state_subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

		# Client for reset the sphere position
		self.client_reset_sphere = self.create_client(SetEntityState, '/gazebo/set_entity_state')

		while not self.client_reset_sphere.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('sphere reset-service not available, waiting...')
		self.request_sphere_reset = SetEntityState.Request()

		# Action-server to change joint position
		self.trajectory_action_client = ActionClient (self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

		# Subcriber topic with the contact sensor
		self.contact_sensor_subscription = self.create_subscription(ContactsState, '/contact_sensor/bumper_link6', self.contact_state_callback, 1)

		self.collision_flag = False
	
	def on_timer_transformation(self):
		
		# We aim with this function to:
		# Look up for the transformation between child_frame and reference frame
		# Determine the pose(position and location of the end effector w.r.t. world frame)

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
			self.robot_x = trans.transform.translation.x
			self.robot_y = trans.transform.translation.y
			self.robot_z = trans.transform.translation.z
			#print ('Translation: [',round (self.robot_x,3), round(self.robot_y,3), round(self.robot_z,3),']')
			
			# Rotation
			self.qx = trans.transform.rotation.x
			self.qy = trans.transform.rotation.y
			self.qz = trans.transform.rotation.z
			self.qw = trans.transform.rotation.w
			#print ('Rotation: in Quaternion [',round (self.qx,3), round(self.qy,3), round(self.qz,3), round(self.qw,3),']')

	def target_state_callback(self, msg):
		# We aim with this function to get the position of the target point (Green sphere)
		sphere_index = msg.name.index('my_sphere') # Get the corret index for the sphere

		# sphere position from Gazebo wrt world frame
		self.pos_sphere_x = msg.pose[sphere_index].position.x 
		self.pos_sphere_y = msg.pose[sphere_index].position.y 
		self.pos_sphere_z = msg.pose[sphere_index].position.z 

	def joint_state_callback(self, msg):
		# We aim with this function to get the state (position) of each joint
		
		# Position:
		self.joint_1_state = msg.position[0]
		self.joint_2_state = msg.position[1]
		self.joint_3_state = msg.position[2]
		self.joint_4_state = msg.position[3]
		self.joint_5_state = msg.position[4]
		self.joint_6_state = msg.position[5]

		# Velocity
		self.joint_1_vel =  msg.velocity[0]
		self.joint_2_vel =  msg.velocity[1]
		self.joint_3_vel =  msg.velocity[2]
		self.joint_4_vel =  msg.velocity[3]
		self.joint_5_vel =  msg.velocity[4]
		self.joint_6_vel =  msg.velocity[5]

	def contact_state_callback(self, msg):
		# We aim with this function to know if the end-effecto touch the ground, contact sensor

		# if self.collision_values is empty [], means there is not collisions
		self.collision_values = msg.states

		if not self.collision_values:
			#self.collision_flag = False
			pass
		else:
			self.collision_flag = True			

	def reset_environment_request(self):

		# Everytime this function is call a request to the Reset the Environment
		# is send i.e. Move the robot to home position and change the
		# sphere location  and waits until get response/confirmation

		self.get_logger().info("Reseting the Environment... ")

		# -----------------------------
		# reset robot position

		home_point_msg = JointTrajectoryPoint()
		home_point_msg.positions     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.velocities    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		home_point_msg.time_from_start = Duration(sec=3)

		joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
		home_goal_msg = FollowJointTrajectory.Goal()
		home_goal_msg.goal_time_tolerance    = Duration(sec=1) 	
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
			self.get_logger().info('Robot in Home Position Without a problem')

		else:
			self.get_logger().info('There was a problem with the accion')

		# -----------------------------
		# reset sphere position

		self.request_sphere_reset.state.name = 'my_sphere'
		self.request_sphere_reset.state.reference_frame = 'world'
		self.request_sphere_reset.state.pose.position.x = random.uniform(-2.0, 2.0)
		self.request_sphere_reset.state.pose.position.y = random.uniform(-2.0, 2.0)
		self.request_sphere_reset.state.pose.position.z = random.uniform(0.1, 2.0)
		
		self.future_sphere_reset = self.client_reset_sphere.call_async(self.request_sphere_reset)

		self.get_logger().info('Reseting sphere to new position...')

		rclpy.spin_until_future_complete(self, self.future_sphere_reset)

		sphere_service_response = self.future_sphere_reset.result()
		
		if sphere_service_response.success:
			self.get_logger().info("Sphere Moved to a New Possiton Success")

		else:
			self.get_logger().info("Sphere Reset Request failed")

		# --------------------------------------
		self.get_logger().info("Environment Reset Success ")

	def action_generator_sample(self):
		# This function just generate random values in radians, 
		# to later be passed as desire values for each joint

		#angle_j_1 = random.uniform(-6.2832, 6.2832)  # values in degrees [-360, 360]
		#angle_j_2 = random.uniform(-1.5708, 1.5708)  # values in degrees [- 90,  90]
		#angle_j_3 = random.uniform(-2.4434, 2.4434)  # values in degrees [-140, 140]
		#angle_j_4 = random.uniform(-6.2832, 6.2832)  # values in degrees [-360, 360]	
		#angle_j_5 = random.uniform(-6.2832, 6.2832)  # values in degrees [-360, 360]	
		#angle_j_6 = random.uniform(-6.2832, 6.2832)  # values in degrees [-360, 360]

		angle_j_1 = random.uniform( 0.000000, 6.28319)   # values in degrees [   0,  360]
		angle_j_2 = random.uniform(-0.610865, 0.610865)	 # values in degrees [- 35,  35]
		angle_j_3 = random.uniform(-0.785398, 0.785398)  # values in degrees [- 45,  45]
		angle_j_4 = random.uniform(-0.785398, 0.785398)	 # values in degrees [- 45,  45]
		angle_j_5 = random.uniform(-0.785398, 0.785398)	 # values in degrees [- 45,  45]
		angle_j_6 = random.uniform(-0.785398, 0.785398)  # values in degrees [- 45,  45]

		return [angle_j_1, angle_j_2, angle_j_3, angle_j_4, angle_j_5, angle_j_6]

	def action_step_service(self, action_values):

		# This function every time it is called passes the goal (desire position of each joint) 
		# to the action-client to execute the trajectory
		

		point_msg = JointTrajectoryPoint()
		point_msg.positions     = action_values
		point_msg.velocities    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		point_msg.time_from_start = Duration(sec=5)


		joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
		goal_msg = FollowJointTrajectory.Goal()
		goal_msg.goal_time_tolerance    = Duration(sec=1) 	# goal_time_tolerance allows some leeway in time, so that the trajectory goal can still
															# succeed even if the joints reach the goal some time after the precise end time of the trajectory.
		goal_msg.trajectory.joint_names = joint_names
		goal_msg.trajectory.points      = [point_msg]

		self.get_logger().info('Waiting for action server...')
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
			self.get_logger().info('Action Completed without a problem')

		else:
			self.get_logger().info('There was a problem with the accion')

		
	def distance_calculator(self):
		# Calculate the distace between the link 6 and the sphere (target point) and get the reward
		# i.e. Calculate the eucladian distance between the link6(end effector) and  shpere (target point)
		
		try:
			robot_end_position    = np.array((self.robot_x, self.robot_y, self.robot_z))
			target_point_position = np.array((self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z))
			
		except: 
			self.get_logger().info('could not calculate the distance yet, trying again...')
			return	

		else:
			distance = np.linalg.norm(robot_end_position - target_point_position)
			return distance

	def reward_calculator(self):
		# I aim with this function to get the reward value. Convert the distance in reward
		
		distance = self.distance_calculator()

		if distance <= 0.05:
			self.get_logger().info('Goal Reached')
			reward_d = 100
			done = True
		else:
			if self.collision_flag:
				self.get_logger().info('Collision link 6')
				reward_d = -100
				done = 'Collision'
				self.collision_flag = False 
			else:
				reward_d = -1
				done = False

		return reward_d, done


	def state_space (self):

		try:
			s = np.array([self.robot_x, self.robot_y, self.robot_z, 
						  self.joint_1_state, self.joint_2_state, self.joint_3_state, self.joint_4_state, self.joint_5_state, self.joint_6_state,
						  self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z
						 ])
		except:
			self.get_logger().info('--------------------node not ready yet, getting values--------------------')
			return

		else:
			return s