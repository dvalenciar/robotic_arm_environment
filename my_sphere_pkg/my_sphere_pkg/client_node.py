'''
Author: David Valencia
Date: 12 / 08 /2021
Describer: 
		   Client Node 
		   
		   This script is a client node. I use this to change the sphere's position in Gazebo.
		   The server is '/gazebo/set_entity_state', and it runs automatically when gazebo stars
		   because I include the gazebo_ros_state plugin in my world file.

		   This client sends a request to the service to change the position on the sphere. 
		   
		   Basically, it sends the position in X, Y, Z where I want to put the sphere. For now,  
		   every time that the node is running sent a random position and wait for the confirmation. 
		
		   Executable name in the setup file: my_client_node

'''

import sys
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState

import random


class MyNodeClient(Node):

	def __init__(self):

		super().__init__('my_client_sphere_node_position')


		self.client_ = self.create_client(SetEntityState, '/gazebo/set_entity_state')

		# Check if the a service is available
		
		while not self.client_.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')

		self.req = SetEntityState.Request()



	def send_request(self):
		
		self.req.state.name = 'my_sphere'
		self.req.state.reference_frame = 'world'
		self.req.state.pose.position.x = random.uniform(-2.0, 2.0)
		self.req.state.pose.position.y = random.uniform(-2.0, 2.0)
		self.req.state.pose.position.z = random.uniform(0.1, 2.0)
		
		# Future is a value that indicates whether the call and response is finished, after sending a request to a service

		self.future = self.client_.call_async(self.req)


def main(args=None):

	rclpy.init(args=args) 

	node_client = MyNodeClient()
	node_client.send_request()


	# See if the service has replied
	
	while rclpy.ok():
		
		rclpy.spin_once(node_client)

		if node_client.future.done():

			# Get response from service 
			try:
				response = node_client.future.result()
				
			except Exception as e:
				node_client.get_logger().info('Service call failed %r' % (e,))

			else:
				node_client.get_logger().info('Coordinates sent status:%s, Points: X:%f Y:%f Z:%f' % 
										     (response.success, 
										      node_client.req.state.pose.position.x, 
										      node_client.req.state.pose.position.y, 
										      node_client.req.state.pose.position.z))		
			break

	
	node_client.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()