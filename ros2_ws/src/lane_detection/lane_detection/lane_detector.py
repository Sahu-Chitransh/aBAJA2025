import cv2
import numpy as np

def detect_lane_and_steering(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    # Detect lines
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)

    # Draw lines
    overlay = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Simple steering estimation: center of lane vs image center
    left_x = []
    right_x = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_x.append(x1)
                left_x.append(x2)
            elif slope > 0.5:
                right_x.append(x1)
                right_x.append(x2)

    lane_center = width // 2
    if left_x and right_x:
        left_mean = np.mean(left_x)
        right_mean = np.mean(right_x)
        lane_center = int((left_mean + right_mean) / 2)

    # Steering calculation (mocked here)
    steering_offset = (lane_center - width // 2) / (width // 2)
    steering_angle = -steering_offset * 25  # Assume ±25 degrees max steering

    return overlay, steering_angle, steering_offset
