#!/usr/bin/env python3
#import all the necessary libraries
import socket
import re
import numpy as np
import cv2
import os
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image #import message

class RSDSCameraPublisher(Node):
    def __init__(self):
        super().__init__('Camera_Publisher')
        self.get_logger().info("Starting Camera Publisher Node...")

        # Connect to Camera RSDS stream
        self.TCP_IP = "192.168.86.1"
        self.TCP_PORT = 2210
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.TCP_IP, self.TCP_PORT))
            self.get_logger().info(f"Connected to RSDS TCP/IP server at {self.TCP_IP}:{self.TCP_PORT}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to RSDS server: {e}")
            raise e

        # RSDS stream contains two parts - first part is the header and second part is the image frame - detailed explanation is available in the reference manual
        self.expected_header_bytes = 64 # this is the header size - default
        self.bridge = CvBridge()

        # Uncomment the below lines if you want to save the image frames in a folder
        # self.image_dir = "/home/vasanth/Images" # make sure to specify ur folder directory 
        # os.makedirs(self.image_dir, exist_ok=True)

        # edit this based on the topic name and message name - Create a publisher (here Image is the message name and RGBImage is the topic)
        self.publisher_ = self.create_publisher(Image, 'RGBImage', 10)
        self.get_logger().info("Publishing created for topic RGBImage")

        # running the camera at 30fps - change it as per need
        self.timer = self.create_timer(0.0333, self.timer_callback)

    def timer_callback(self):
        try:
            # reading the 64 bytes of header information
            header_data = b""
            while len(header_data) < self.expected_header_bytes:
                packet = self.sock.recv(self.expected_header_bytes - len(header_data))
                if not packet:
                    self.get_logger().error("Socket closed while reading header.")
                    return
                header_data += packet
            
            header_str = header_data.decode("utf-8", errors="ignore")
            pattern = r"\*RSDS\s+(\d+)\s+(\S+)\s+([\d\.]+)\s+(\d+)x(\d+)\s+(\d+)"
            match = re.search(pattern, header_str)
            if not match:
                self.get_logger().error(f"Failed to parse RSDS header:\n{header_str}")
                return

            channel = int(match.group(1))
            sim_time = float(match.group(3))
            width = int(match.group(4))
            height = int(match.group(5))
            img_len = int(match.group(6))

            expected_len = width * height * 3
            if img_len != expected_len:
                self.get_logger().warn(f"Image length mismatch: expected {expected_len}, got {img_len}")
                return

            # reading the image frame
            raw_data = b""
            while len(raw_data) < img_len:
                packet = self.sock.recv(img_len - len(raw_data))
                if not packet:
                    self.get_logger().error("Socket closed while reading image data.")
                    return
                raw_data += packet
            
            image_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))

            # Save image to disk - Uncomment the below lines if you want to save the image frames in a folder
            # filename = f"{self.image_dir}/frame_{int(sim_time * 1000)}.png"
            # success = cv2.imwrite(filename, image_np)
            # if success:
            #    self.get_logger().info(f"Image saved to {filename}")
            # else:
            #    self.get_logger().error(f"Failed to save image to {filename}")

            # Convert and publish
            ros_image = self.bridge.cv2_to_imgmsg(image_np, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"

            self.publisher_.publish(ros_image)
            # self.get_logger().info(f"Published frame at time {sim_time}, channel {channel}") # uncomment if you want a feedback when the publisher is running

        except Exception as e:
            self.get_logger().error(f"Timer callback failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RSDSCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Camera Publisher Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
