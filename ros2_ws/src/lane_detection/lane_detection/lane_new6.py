import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.bridge = CvBridge()
        self.subscriber = self.create_subscription(Image, '/RGBImage', self.image_callback, 10)
        self.curvature_pub = self.create_publisher(Float32, '/lane_curvature', 10)
        self.steering_pub = self.create_publisher(Float32, '/steering_angle', 10)
        self.debug_img_pub = self.create_publisher(Image, '/lane_debug_image', 10)
        self.get_logger().info('LaneDetectorNode Initialized')
        self.prev_left_fit = None
        self.prev_right_fit = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        binary = self.preprocess_image(frame)
        warped, matrix = self.warp_perspective(binary)
        left_fit, right_fit = self.fit_polynomial(warped)

        if left_fit is None and self.prev_left_fit is not None:
            left_fit = self.prev_left_fit
        if right_fit is None and self.prev_right_fit is not None:
            right_fit = self.prev_right_fit

        if left_fit is not None: self.prev_left_fit = left_fit
        if right_fit is not None: self.prev_right_fit = right_fit

        curvature = self.measure_curvature_real(left_fit, right_fit, frame.shape[0])
        steering_angle = self.compute_steering_angle(left_fit, right_fit, frame.shape[1], frame.shape[0])

        if curvature is not None:
            self.curvature_pub.publish(Float32(data=float(curvature)))
        if steering_angle is not None:
            self.steering_pub.publish(Float32(data=float(steering_angle)))

        overlay = self.draw_lane_overlay(frame, warped, left_fit, right_fit, matrix)
        debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self.debug_img_pub.publish(debug_msg)

    def preprocess_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 40, 255))
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        _, binary = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return binary

    def warp_perspective(self, img):
        h, w = img.shape
        src = np.float32([
            [w * 0.45, h * 0.55],
            [w * 0.55, h * 0.55],
            [w * 0.9, h],
            [w * 0.08, h]
        ])
        dst = np.float32([
            [w * 0.2, 0],
            [w * 0.8, 0],
            [w * 0.8, h],
            [w * 0.2, h]
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, matrix, (w, h))
        return warped, matrix

    def fit_polynomial(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None
        return left_fit, right_fit

    def draw_lane_overlay(self, original_img, binary_warped, left_fit, right_fit, matrix):
        h, w = binary_warped.shape
        ploty = np.linspace(0, h-1, h)
        color_warp = np.zeros((h, w, 3), dtype=np.uint8)

        if left_fit is None or right_fit is None:
            return original_img

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        Minv = np.linalg.inv(matrix)
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.5, 0)
        return result

    def measure_curvature_real(self, left_fit, right_fit, y_eval, ym_per_pix=30/720, xm_per_pix=3.7/700):
        if left_fit is None or right_fit is None:
            return None
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / abs(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / abs(2*right_fit[0])
        return (left_curverad + right_curverad) / 2

    def compute_steering_angle(self, left_fit, right_fit, image_width, image_height):
        if left_fit is None or right_fit is None:
            return None
        y_eval = image_height
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2
        vehicle_center = image_width / 2
        offset = lane_center - vehicle_center
        return -offset / image_width * 60

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()