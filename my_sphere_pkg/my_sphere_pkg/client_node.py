'''
Author: David Valencia
Date: 12 / 08 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)
Describer:
           Client Node

           This script is a client node used to change the sphere's position in
           Gazebo (gz-sim / Harmonic).

           Migration note:
             - Gazebo Classic used the service '/gazebo/set_entity_state'
               (gazebo_msgs/SetEntityState), provided by the gazebo_ros_state plugin.
             - Gazebo Harmonic exposes '/world/<world>/set_pose' instead. We reach
               it through the ros_gz_bridge as the ROS service
               '/world/default/set_pose' of type ros_gz_interfaces/srv/SetEntityPose.
               (See config/sphere_bridge.yaml.)

           This client sends a request to teleport the sphere to a random X, Y, Z.

           Executable name in the setup file: my_client_node
'''

import random

import rclpy
from rclpy.node import Node

from ros_gz_interfaces.srv import SetEntityPose


class MyNodeClient(Node):

    def __init__(self):

        super().__init__('my_client_sphere_node_position')

        self.client_ = self.create_client(SetEntityPose, '/world/default/set_pose')

        # Check if the service is available
        while not self.client_.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.req = SetEntityPose.Request()

    def send_request(self):

        # Target the sphere model by name
        self.req.entity.name = 'my_sphere'
        self.req.entity.type = self.req.entity.MODEL

        # Random goal position
        self.req.pose.position.x = random.uniform(-2.0, 2.0)
        self.req.pose.position.y = random.uniform(-2.0, 2.0)
        self.req.pose.position.z = random.uniform(0.1, 2.0)
        self.req.pose.orientation.w = 1.0

        # Future indicates whether the call and response is finished
        self.future = self.client_.call_async(self.req)


def main(args=None):

    rclpy.init(args=args)

    node_client = MyNodeClient()
    node_client.send_request()

    # See if the service has replied
    while rclpy.ok():

        rclpy.spin_once(node_client)

        if node_client.future.done():

            try:
                response = node_client.future.result()

            except Exception as e:
                node_client.get_logger().info('Service call failed %r' % (e,))

            else:
                node_client.get_logger().info(
                    'Coordinates sent status:%s, Points: X:%f Y:%f Z:%f' %
                    (response.success,
                     node_client.req.pose.position.x,
                     node_client.req.pose.position.y,
                     node_client.req.pose.position.z))
            break

    node_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
