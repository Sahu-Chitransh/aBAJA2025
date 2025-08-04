import cv2
import numpy as np
import math
import warnings

class RankWarning(UserWarning):
    pass

warnings.simplefilter('ignore', RankWarning)

initial_steering_angle = 0  # Set initial steering angle as straight (0 degrees)
smoothed_angle = 0          # For smoothing sudden fluctuations
alpha = 0.2                 # Smoothing factor (0.0 = no smoothing, 1.0 = instant)



prev_left_fit_average = prev_right_fit_average = None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.85)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left_fit_average, prev_right_fit_average
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x2 - x1) < 10:
            continue  # Skip vertical/near-vertical lines to avoid polyfit warning

        #x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        prev_left_fit_average = left_fit_average
        prev_right_fit_average = right_fit_average
    except:
        left_line = make_coordinates(image, prev_left_fit_average)
        right_line = make_coordinates(image, prev_right_fit_average)
    left_angle = np.degrees(np.arctan(left_fit_average[0]))
    right_angle = np.degrees(np.arctan(right_fit_average[0]))
    average_angle = (left_angle + right_angle) / 2
    return np.array([left_line, right_line]), average_angle


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 100), 12)
    return line_image

def use_sliding_window(image):
    frame = cv2.resize(image, (640, 480))
    tl, bl, tr, br = (150, 387), (70, 472), (500, 380), (538, 472)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(frame, matrix, (640, 480))
    hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
    lower_white, upper_white = (0, 0, 200), (180, 55, 255)
    lower_yellow, upper_yellow = (15, 80, 120), (40, 255, 255)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    msk = mask.copy()
    y = 472
    while y > 0:
        left_img = msk[y - 40:y, left_base - 50:left_base + 50]
        right_img = msk[y - 40:y, right_base - 50:right_base + 50]
        contours_left, _ = cv2.findContours(left_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_left:
            M = cv2.moments(cnt)
            if M["m00"]:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - 50 + cx
        contours_right, _ = cv2.findContours(right_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_right:
            M = cv2.moments(cnt)
            if M["m00"]:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - 50 + cx
        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), 255, 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), 255, 2)
        y -= 40
    return msk

def curvature_is_high(lines):
    if lines is None:
        return True
    threshold = 0.5
    curvatures = []
    for x1, y1, x2, y2 in lines:
        slope = abs((y2 - y1) / (x2 - x1 + 1e-5))
        curvatures.append(slope)
    return any(s < threshold for s in curvatures)


# NEW FUNCTION
def calculate_angle(x1, y1, x2, y2):
    radians = math.atan2((y2 - y1), (x2 - x1))
    degrees = math.degrees(radians)
    return degrees

cap = cv2.VideoCapture("LaneVideo.mp4")

angle=0

focal_length = 500  # Approximate focal length in pixels for 640px width
steering_angle = 0  # Initial steering angle assumed 0°
Kp = 1.0  # Proportional constant (adjust as needed)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    if lines is not None:
        averaged_lines, angle = average_slope_intercept(frame, lines)
    else:
        averaged_lines = None

    if averaged_lines is not None and not curvature_is_high(averaged_lines):
        # --- Hough Transform path ---
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)

        # ----- Lane angle from center -----
        left_line, right_line = averaged_lines
        # Use midpoints instead of just bottom x-values
        left_x = (left_line[0] + left_line[2]) // 2
        right_x = (right_line[0] + right_line[2]) // 2
        lane_center = (left_x + right_x) // 2

        image_center = frame.shape[1] // 2
        offset = lane_center - image_center
        print(f"Left line: {left_line}, Right line: {right_line}")
        print(f"Lane center: {lane_center}, Image center: {image_center}, Offset: {offset}")
        center_angle = math.degrees(math.atan(offset / focal_length))
        smoothed_angle = (1 - alpha) * smoothed_angle + alpha * center_angle
        required_steering_angle = initial_steering_angle + smoothed_angle

        steering_angle = Kp * center_angle  # Update based on proportional control
        print(f"[Hough] Raw Angle: {center_angle:.2f}°, Smoothed: {smoothed_angle:.2f}°, Steering: {required_steering_angle:.2f}°")

        # Display on video
        cv2.putText(combo_image, f"Angle (Smoothed): {smoothed_angle:.2f} deg", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    else:
        # --- Sliding Window path ---
        msk = use_sliding_window(frame)
        msk_resized = cv2.resize(msk, (frame.shape[1], frame.shape[0]))
        combo_image = cv2.addWeighted(frame, 0.9, cv2.cvtColor(msk_resized, cv2.COLOR_GRAY2BGR), 1, 1)

        # ----- Lane center for sliding window -----
        histogram = np.sum(msk[msk.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        lane_center = (left_base + right_base) // 2
        image_center = frame.shape[1] // 2
        offset = lane_center - image_center
        center_angle = math.degrees(math.atan(offset / focal_length))
        steering_angle = Kp * center_angle
        print(f"[Sliding Window] Steering Angle: {steering_angle:.2f}° (based on center)")

        # Display on video
        cv2.putText(combo_image, f"Angle (Center): {center_angle:.2f} deg", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Result', combo_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()