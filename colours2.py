import logging

import cv2
import numpy as np
import pytesseract


def find_white_rectangles_in_binary(image, rectangular_contours):
    white_rectangles = []
    for cnt in rectangular_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y:y + h, x:x + w]
        white_ratio = cv2.countNonZero(roi) / (roi.shape[0] * roi.shape[1])
        if white_ratio > 0.90:
            white_rectangles.append(cnt)
    return white_rectangles


def count_white_divisions(binary_image, rectangular_contours, w_smallest):
    white_divisions_count = 0

    for contour in rectangular_contours:
        x, y, w, h = cv2.boundingRect(contour)
        divisions = round(w / w_smallest)

        for j in range(divisions):
            x_divided = int(x + (j * w_smallest))
            w_divided = int(w / divisions)
            division_roi = binary_image[y:y + h, x_divided:x_divided + w_divided]

            # Calculate the percentage of white area in the division
            total_pixels = division_roi.shape[0] * division_roi.shape[1]
            white_pixels = cv2.countNonZero(division_roi)
            white_percentage = (white_pixels / total_pixels) * 100

            # If the white percentage is >= 90%, increment the count
            if white_percentage >= 90:
                white_divisions_count += 1

    return white_divisions_count


from ocr_with_multiple_psms_first import ocr_with_multiple_psms_first

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
# Load the image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to correct the orientation of the tile
def correct_orientation(roi):
    coords = np.column_stack(np.where(roi > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = roi.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated



def find_rectangular_contours_with_fixed_threshold(image, target_num_rectangles):
    #image = cv2.resize(image, (1200, 1200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    numbers_at_positions = []  # List to keep track of numbers found at each position

    for thresh_value in range(150, 220, 1):
        _, thresh = cv2.threshold(equalized, thresh_value, 255, cv2.THRESH_BINARY)
        logger.info(f'Currently at : {thresh_value}')
        blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
        median = cv2.medianBlur(thresh, 7)
        contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        filtered_contours = [cnt for cnt in contours if
                             10000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0]) / 2]

        rectangular_contours = []
        heights = []
        for cnt in filtered_contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:  # Ensure the contour is rectangular
                x, y, w, h = cv2.boundingRect(approx)
                heights.append(h)
                rectangular_contours.append(cnt)

        # Sort rectangular contours based on area and select the top 10
        rectangular_contours = sorted(rectangular_contours, key=cv2.contourArea, reverse=True)[:10]
        if len(rectangular_contours) != target_num_rectangles:
            continue  # Skip to the next threshold if the count doesn't match the target
            # Check if all heights are approximately the same
        height_threshold = 100  # You can adjust this threshold as needed
        if max(heights) - min(heights) > height_threshold:
            logger.error("Rectangles do not have approximately the same height.")
            continue
        # Find the smallest contour
        smallest_contour = min(rectangular_contours, key=cv2.contourArea)
        x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
        # Draw rectangles on the image
        # Track numbers found in the current threshold run
        current_run_numbers = []

        i = 0
        while i < len(rectangular_contours):
            approx = rectangular_contours[i]
            x, y, w, h = cv2.boundingRect(approx)
            divisions = round((w / w_smallest))
            if divisions < 1:
                divisions = 1

            j = 0
            while j < divisions:
                x_divided = int(x + (j * w_smallest))
                w_divided = int(w / divisions)
                upper_half_roi = blurred[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
                kernel = np.ones((3, 3), np.uint8)
                upper_half_roi = cv2.dilate(upper_half_roi, kernel)
                colored_roi = image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
                gray_roi = equalized[y:y + int(h // 1.5), x_divided:x_divided + w_divided]

                upper_half_roi = gray_roi

                # Apply OCR to the division
                psms_to_try = [3,4,5,6, 7, 8, 9,10,11,12,13]
                text_blurred = ocr_with_multiple_psms_first(upper_half_roi, psms_to_try)
                text_gray = ocr_with_multiple_psms_first(gray_roi, psms_to_try)

                if text_gray.strip().isdigit() and 1 <= int(text_gray) <= 13:
                    if text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
                        number = max(int(text_gray), int(text_blurred))
                    else:
                        number = int(text_gray)
                elif text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
                    number = int(text_blurred)
                else:
                    number = "Jolly"

                current_run_numbers.append(number)
                j += 1

            i += 1

        # Append current run numbers to the tracking list
        numbers_at_positions.append(current_run_numbers)
        logger.info(current_run_numbers)

    # Determine the predominant numerical value at each position
    if numbers_at_positions:
        positions_length = max(len(numbers) for numbers in numbers_at_positions)
        predominant_numbers = []
        for position in range(positions_length):
            count_dict = {}
            valid_number_seen = False
            for numbers in numbers_at_positions:
                if position < len(numbers):
                    number = numbers[position]
                    if number != "Jolly":
                        valid_number_seen = True
                        if number in count_dict:
                            count_dict[number] += 1
                        else:
                            count_dict[number] = 1
            if valid_number_seen and count_dict:
                predominant_number = max(count_dict, key=count_dict.get)
                predominant_numbers.append(predominant_number)
            else:
                predominant_numbers.append("Jolly")
        return predominant_numbers

    return None


# Load the image
#merge pentru remiacasaV2,remipebune
image_path = 'MainApp/poze/remipebune.jpg'
image = cv2.imread(image_path)

# Set the target number of rectangles to find
target_num_rectangles = 3 # You can adjust this value

# Find the smallest contour
result = find_rectangular_contours_with_fixed_threshold(image, target_num_rectangles)
logger.info(result)
