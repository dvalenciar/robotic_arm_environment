'''
Author: David Valencia
Date: 12 / 10 /2021
Describer: 

		I use this file to run my custom environment easier, the whole body of the environment is in 
		the file main_environmet. py, but here I used to import that file and run the ROS node more organized
		
		To use this script easily I need to launch first:
		
		my_environment.launch.py
		
				
		Executable name in the setup file: run_environment
'''

import time
import rclpy
from .main_environment import MyEnvironmentNode


def main(args=None):

	rclpy.init(args=args)
	run_env_node = MyEnvironmentNode()
	rclpy.spin_once(run_env_node)


	num_episodes = 3
	episonde_horizont = 5


	for episode in range (num_episodes):		
		for step in range (episonde_horizont):
			print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

			action = run_env_node.action_generator_sample() # generate a sample action vector
			run_env_node.action_step_service(action) # take the action
			
			reward, done  = run_env_node.reward_calculator()
			state  = run_env_node.state_space()

			if done: 
				# if done is TRUE means the end-effector reach to goal and environmet will reset
				print (f'Goal Reach, Episode ends after {step+1} steps')
				break
			
			time.sleep(1.0)
		
		print (f'Episode {episode+1} Ended')
		
		run_env_node.reset_environment_request()					
		time.sleep(2.0)
		step = 0

	print ("Total num of episode completed, Exiting ....")
	#run_env_node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
