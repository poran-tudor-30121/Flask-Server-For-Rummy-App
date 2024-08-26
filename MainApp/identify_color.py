import cv2
import numpy as np

def identify_color(hsv_image):
    # Define color ranges for red, yellow, blue, and black
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])  # First range for red

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])  # Second range for red

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_blue1 = np.array([90, 50, 50])  # First range for blue
    upper_blue1 = np.array([110, 255, 255])

    lower_blue2 = np.array([110, 50, 50])  # Second range for blue
    upper_blue2 = np.array([130, 255, 255])

    lower_black1 = np.array([0, 0, 0])  # First range for black
    upper_black1 = np.array([180, 50, 50])  # Adjusted upper bound for black

    lower_black2 = np.array([0, 0, 30])  # Second range for black
    upper_black2 = np.array([180, 255, 70])  # Adjusted upper bound for black

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_blue1 = cv2.inRange(hsv_image, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(hsv_image, lower_blue2, upper_blue2)
    mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
    mask_black1 = cv2.inRange(hsv_image, lower_black1, upper_black1)
    mask_black2 = cv2.inRange(hsv_image, lower_black2, upper_black2)
    mask_black = cv2.bitwise_or(mask_black1, mask_black2)

    # Calculate the sum of pixel values in each color's mask
    sum_red = np.sum(mask_red)
    sum_yellow = np.sum(mask_yellow)
    sum_blue = np.sum(mask_blue)
    sum_black = np.sum(mask_black)

    # Classify color based on the maximum sum
    color_mapping = {sum_red: 'Red', sum_yellow: 'Yellow', sum_blue: 'Blue', sum_black: 'Black'}
    max_sum = max(sum_red, sum_yellow, sum_blue, sum_black)

    return color_mapping[max_sum]

# Example usage:
# Assuming you have an HSV image (hsv_image) obtained from cv2.cvtColor() conversion
# color_detected = identify_color(hsv_image)
# print("Detected color:", color_detected)
