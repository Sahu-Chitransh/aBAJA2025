import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import math
from lane_detection.lane import (
    canny,
    region_of_interest,
    average_slope_intercept,
    display_lines,
    use_sliding_window,
    curvature_is_high
)

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/RGBImage', self.image_callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/lane_info', 10)
        self.visual_pub = self.create_publisher(Image, '/lane_viz', 10)

        self.smoothed_angle = 0
        self.alpha = 0.2
        self.Kp = 1.0
        self.initial_steering_angle = 0
        self.focal_length = 500  # assumed camera focal length

        self.get_logger().info("Lane Detector Node Initialized")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        if lines is not None:
            averaged_lines, _ = average_slope_intercept(frame, lines)
        else:
            averaged_lines = None

        if averaged_lines is not None and not curvature_is_high(averaged_lines):
            left_line, right_line = averaged_lines
            left_x = (left_line[0] + left_line[2]) // 2
            right_x = (right_line[0] + right_line[2]) // 2
            lane_center = (left_x + right_x) // 2
            image_center = frame.shape[1] // 2
            offset = lane_center - image_center
            center_angle = math.degrees(math.atan(offset / self.focal_length))
            self.smoothed_angle = (1 - self.alpha) * self.smoothed_angle + self.alpha * center_angle
            steering_angle = self.initial_steering_angle + self.smoothed_angle
        else:
            msk = use_sliding_window(frame)
            histogram = np.sum(msk[msk.shape[0] // 2:, :], axis=0)
            midpoint = histogram.shape[0] // 2
            left_base = np.argmax(histogram[:midpoint])
            right_base = np.argmax(histogram[midpoint:]) + midpoint
            lane_center = (left_base + right_base) // 2
            image_center = frame.shape[1] // 2
            offset = lane_center - image_center
            center_angle = math.degrees(math.atan(offset / self.focal_length))
            steering_angle = self.Kp * center_angle
        '''
        # Publish steering angle and offset
        if not isinstance(steering_angle, (float, int)) or not isinstance(offset, (float, int)):
            self.get_logger().warn(f"Invalid data — skipping publish: steering_angle={steering_angle}, offset={offset}")
            return

        msg_out = Float32MultiArray()
        #msg_out.data = [steering_angle, offset]
        msg_out.data = [float(steering_angle), float(offset)]
        self.pub.publish(msg_out)
        '''
        try:
            angle = float(steering_angle)
            lane_offset = float(offset)

            if not np.isfinite(angle) or not np.isfinite(lane_offset):
                raise ValueError("Non-finite values")

            msg_out = Float32MultiArray()
            msg_out.data = [angle, lane_offset]
            self.pub.publish(msg_out)

        except (TypeError, ValueError) as e:
            self.get_logger().warn(f"🚫 Skipped publishing lane_info: {e} | angle={steering_angle}, offset={offset}")

        # Visualization image
        if averaged_lines is not None:
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)
        else:
            msk_resized = cv2.resize(msk, (frame.shape[1], frame.shape[0]))
            combo_image = cv2.addWeighted(frame, 0.9, cv2.cvtColor(msk_resized, cv2.COLOR_GRAY2BGR), 1, 1)

        visual_msg = self.bridge.cv2_to_imgmsg(combo_image, encoding='bgr8')
        visual_msg.header = msg.header
        self.visual_pub.publish(visual_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
