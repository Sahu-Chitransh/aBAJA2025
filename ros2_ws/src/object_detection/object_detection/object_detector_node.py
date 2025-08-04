import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class YOLOv8ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        self.subscription = self.create_subscription(
            Image,
            '/RGBImage',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            Image,
            '/detected_image',
            10
        )

        self.bridge = CvBridge()
        self.model = YOLO("/home/aBAJA2025/ros2_ws/src/object_detection/object_detection/yolov8m.pt")  # Change path if needed

        self.get_logger().info("YOLOv8 Object Detector Node Initialized")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")
            return

        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]

        # Draw bounding boxes and class labels
        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            class_name = self.model.names[cls_id]
            label = f"{class_name} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            self.get_logger().info(f"Detected: {label}")

        # Show the frame using OpenCV
        cv2.imshow("YOLOv8 Detection", frame)
        cv2.waitKey(1)  # Required for OpenCV GUI to update

        # Publish the detected image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()  # Clean up OpenCV windows
    rclpy.shutdown()

if __name__ == '__main__':
    main()
