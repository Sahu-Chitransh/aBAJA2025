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
        self.overlay_pub = self.create_publisher(Image, '/lane_overlay', 10)
        self.get_logger().info('LaneDetectorNode Initialized')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        roi = self.region_of_interest(edges)
        warped, matrix = self.warp_perspective(roi)
        left_fit, right_fit = self.fit_polynomial(warped)

        curvature = self.measure_curvature_real(left_fit, right_fit, frame.shape[0])
        steering_angle = self.compute_steering_angle(left_fit, right_fit, frame.shape[1], frame.shape[0])

    # --- Overlay for visualization ---
        lane_overlay = self.draw_lane_overlay(frame, warped, left_fit, right_fit, matrix)
        annotated = lane_overlay.copy()

        if curvature is not None:
            cv2.putText(annotated, f"Curvature: {curvature:.2f}m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if steering_angle is not None:
            cv2.putText(annotated, f"Steering: {steering_angle:.2f} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

    # Publish scalar values
        if curvature is not None:
            self.curvature_pub.publish(Float32(data=float(curvature)))
        if steering_angle is not None:
            self.steering_pub.publish(Float32(data=float(steering_angle)))


    def region_of_interest(self, img):
        h, w = img.shape
        mask = np.zeros_like(img)
        polygon = np.array([[
            (int(0.1 * w), h),
            (int(0.4 * w), int(0.6 * h)),
            (int(0.6 * w), int(0.6 * h)),
            (int(0.9 * w), h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def warp_perspective(self, img):
        h, w = img.shape
        src = np.float32([
            [w * 0.45, h * 0.63],
            [w * 0.55, h * 0.63],
            [w * 0.9, h],
            [w * 0.1, h]
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

    def measure_curvature_real(self, left_fit, right_fit, y_eval, ym_per_pix=30/720, xm_per_pix=3.7/700):
        if left_fit is None or right_fit is None:
            return None
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
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
        return -offset / image_width * 60  # steering angle in degrees (approx)
    
    def draw_lane_overlay(self, original_img, binary_warped, left_fit, right_fit, matrix):
        h, w = binary_warped.shape
        ploty = np.linspace(0, h - 1, h)
        color_warp = np.zeros((h, w, 3), dtype=np.uint8)

        if left_fit is None or right_fit is None:
            return original_img

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        Minv = np.linalg.inv(matrix)
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.5, 0)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
