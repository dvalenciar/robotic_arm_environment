'''
Author: David Valencia
Date: 11 / 08 / 2021  (migrated to ROS 2 Jazzy + Gazebo Harmonic, 2026-06)

Describer:
           This script reads the position of the sphere from Gazebo and
           republishes it as a Marker (sphere) in the topic 'marker_position',
           used to visualize the goal in RViz.

           Migration note:
             - Gazebo Classic published model poses on '/gazebo/model_states'
               (gazebo_msgs/ModelStates).
             - In Gazebo Harmonic the sphere model carries a PosePublisher plugin
               (see models/sdf/sphere_goal/model.sdf) that emits its pose on
               '/model/my_sphere/pose' (gz.msgs.Pose) at 20 Hz, bridged to ROS
               as a geometry_msgs/Pose. (See config/sphere_bridge.yaml.)

           Executable name in the setup file: reader_mark_node
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker


class MyNode(Node):

    def __init__(self):

        super().__init__('node_sphere_position_mark')

        self.marker_publisher = self.create_publisher(Marker, 'marker_position', 10)
        self.pose_subscription = self.create_subscription(
            Pose, '/model/my_sphere/pose', self.pose_listener_callback, 10)

    def pose_listener_callback(self, msg):

        pos = msg.position

        # Publish a Marker with the sphere's position (for RViz)
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.id = 0
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        marker.pose.position.x = pos.x
        marker.pose.position.y = pos.y
        marker.pose.position.z = pos.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_publisher.publish(marker)


def main(args=None):

    rclpy.init(args=args)

    node = MyNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()