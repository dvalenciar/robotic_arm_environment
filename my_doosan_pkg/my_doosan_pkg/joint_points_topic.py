'''

Author: David Valencia
Date: 26 / 08 /2021

Describer:  This script publishes the position "angles" of each joint under 
			the topic /joint_trajectory_controller/joint_trajectory
			
            I need to run first the my_doosan_controller.launch  in order to load and start the controllers
            update: also can lauch my launch enviroment file

            in simple terms, move the robot joints to the desired position using a topic and the controller

            Executable name in the setup file: trajectory_points_topic     			
'''

import rclpy

from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory , JointTrajectoryPoint


class TrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('topic_desired_trajectory_publisher_node')
        
        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.trajectory_publisher = self.create_publisher(JointTrajectory,"/joint_trajectory_controller/joint_trajectory", 10)


    def timer_callback(self):

        # creating a point
        goal_positions = [-0.5,-0.5,-0.5,0.5,0.5,-0.5]
        
        point_msg = JointTrajectoryPoint()
        point_msg.positions = goal_positions
        point_msg.time_from_start = Duration(sec=2)


        # adding newly created point into trajectory message
        joints = ['joint1','joint2','joint3','joint4','joint5','joint6']

        my_trajectory_msg = JointTrajectory()
        my_trajectory_msg.joint_names = joints
        my_trajectory_msg.points.append(point_msg)
        
        self.trajectory_publisher.publish(my_trajectory_msg)


def main(args=None):

    rclpy.init(args=args)
    joint_trajectory_object = TrajectoryPublisher()

    rclpy.spin(joint_trajectory_object)
    
    joint_trajectory_object.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

