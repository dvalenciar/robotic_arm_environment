'''
Author: David Valencia
Date     : 12 / 10 /2021
Modified : 07 /04  /2022


Describer: 

		I use this file to run my main environment easier, the whole body of the environment is in 
		the file main_rl_environmet.py, but here I import that file and run the ROS node more organized

		This file just generates a vector of random action and moves the robot until the number of episodes 
		and the number of steps is completed, nothing more
		
		To use this script I need to launch first:
			my_environment.launch.py

		Executable name of this file in the setup file: run_environment
'''

import time
import rclpy
from .main_rl_environment import MyRLEnvironmentNode


def main(args=None):

	rclpy.init(args=args)
	run_env_node = MyRLEnvironmentNode()
	rclpy.spin_once(run_env_node)

	num_episodes = 3
	episonde_horizont = 5

	for episode in range (num_episodes):

		run_env_node.reset_environment_request()					
		time.sleep(2.0)
		step = 0
		
		for step in range (episonde_horizont):
			print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

			action = run_env_node.generate_action_funct() # generate a sample action vector
			run_env_node.action_step_service(action) # take the action		
			reward, done  = run_env_node.calculate_reward_funct()
			state  = run_env_node.state_space_funct()

			if done == True: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
				print (f'Goal Reach, Episode ends after {step+1} steps')
				break

			time.sleep(1.0)
			
		print (f'Episode {episode+1} Ended')
		

	print ("Total num of episode completed, Exiting ....")
	rclpy.shutdown()
	

if __name__ == '__main__':
	main()
