#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np

from lane_detection.lane_detector import detect_lane_and_steering

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.get_logger().info("Lane Detector Node Initialized")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/RGBImage',
            self.image_callback,
            10)

        self.overlay_publisher = self.create_publisher(Image, '/lane_overlay_image', 10)
        self.steering_angle_pub = self.create_publisher(Float32, '/steering_angle', 10)
        self.steering_offset_pub = self.create_publisher(Float32, '/steering_offset', 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        overlay, angle, offset = detect_lane_and_steering(frame)

        # Publish overlay image
        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
        out_msg.header = msg.header
        self.overlay_publisher.publish(out_msg)

        # Publish steering angle and offset
        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.steering_angle_pub.publish(angle_msg)

        offset_msg = Float32()
        offset_msg.data = float(offset)
        self.steering_offset_pub.publish(offset_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Lane Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
