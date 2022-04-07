'''

Author: David Valencia
Date: 07/ 04 /2022

Describer: 
		
		This script runs a specific number of episodes where random actions are performed. 
		The robot will move using those actions and the state, reward and next state (the state after performing the actions)
		are saved in a txt file (one for each variable). This script imports the main environment automatically. 

		Executable name in the setup file:data_collection
'''

import sys
import time
import rclpy
import numpy as np
from .main_rl_environment import MyRLEnvironmentNode	


num_episodes = 20
episonde_horizont = 3

current_state_vector = []
next_state_vector    = []
reward_vector        = []
action_vectors       = []

store_path = '/home/david/ros2_ws/src/robotic_arm_environment/data/'



def write_data_function(path, st,  act, st_1, rew):

	np.savetxt(path + "/current_state.txt",st,  fmt='%4f')
	np.savetxt(path + "/action_vector.txt",act, fmt='%4f')
	np.savetxt(path + "/next_state.txt",   st_1,fmt='%4f')
	np.savetxt(path + "/rew_vector.txt",   rew, fmt='%f')


def collector_fuction(data_collector):

	for episode in range (num_episodes):

		data_collector.reset_environment_request()
		time.sleep(2.0)
		step = 0
	
		for step in range (episonde_horizont):

			print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

			current_state = data_collector.state_space_funct() # get the current state				
			action_sample = data_collector.generate_action_funct() # generate a random action vector
			
			data_collector.action_step_service(action_sample) # take the action
			
			reward, done = data_collector.calculate_reward_funct() # get the reward
			next_state   = data_collector.state_space_funct()      # get the state after the taking the actions
			
			if done == True: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
				print (f'Goal Reach, Episode ends after {step+1} steps')
				break

			if current_state == None:
				# Just to be sure that the values are arriving and to not get None values
				print("None value")
				pass
			else:
				#store the values
				current_state_vector.append(current_state)	
				action_vectors.append(action_sample)
				next_state_vector.append(next_state)
				reward_vector.append(reward)
				write_data_function(store_path, current_state_vector, action_vectors, next_state_vector ,reward_vector)

			time.sleep(1.0)

		print (f'Episode {episode+1} Ended')
	
	print ("Total num of episode completed, Exiting ....")



def main (args=None):

	rclpy.init(args=args)
	
	try:
		data_collector_node = MyRLEnvironmentNode()
		rclpy.spin_once(data_collector_node)
		collector_fuction(data_collector_node)

	except KeyboardInterrupt:
		print('ROS2 server stopped cleanly')
	except BaseException:
		print('exception in server:', file=sys.stderr)
		raise		

	finally:
		rclpy.shutdown()
		print("Good bye")
 

if __name__ == '__main__':
	main()
