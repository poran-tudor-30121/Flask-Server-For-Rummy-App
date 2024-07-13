import asyncio
import logging
from collections import Counter

import cv2
import numpy as np
import pytesseract
import threading
from identify_color import identify_color
from ocr_with_multiple_psms import ocr_with_multiple_psms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_contour_height(contour):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    return h  # Return the height of the bounding rectangle


def group_contours_by_height(contours, tolerance):
    # Sort contours by height
    contours_sorted = sorted(contours, key=get_contour_height)

    grouped_contours = []
    current_group = []

    # Group contours by height
    for contour in contours_sorted:
        if not current_group or abs(get_contour_height(contour) - get_contour_height(current_group[0])) <= tolerance:
            current_group.append(contour)
        else:
            grouped_contours.append(current_group)
            current_group = [contour]

    if current_group:
        grouped_contours.append(current_group)

    # Return the group with the most contours
    return max(grouped_contours, key=len, default=[])

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")

    return cv2.LUT(image, table)

# Define a function to draw rectangles on the original image
def draw_rectangles(image, rectangles):
    for rect in rectangles:
        x, y, w, h =  cv2.boundingRect(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def find_color(original_roi):
    hsv_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2HSV)
    color = identify_color(hsv_roi)
    return color

async def process_divisions_async(rectangular_contours, original_image, median_adaptive, numbers_at_positions, logger):
    current_run_numbers_and_colors = []
    smallest_contour = min(rectangular_contours, key=cv2.contourArea)
    x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
    i = 0

    while i < len(rectangular_contours):
        approx = rectangular_contours[i]
        x, y, w, h = cv2.boundingRect(approx)
        divisions = round(w / w_smallest)
        if divisions < 1:
            divisions = 1

        j = 0
        while j < divisions:
            x_divided = int(x + (j * w_smallest))
            w_divided = int(w / divisions)

            # Region of interest (ROI) for thresholded and original images
            thresh_roi = median_adaptive[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            thresh_roi = cv2.resize(thresh_roi, (300, 300))
            kernel = np.ones((3, 3), np.uint8)
            thresh_roi = cv2.dilate(thresh_roi, kernel)

            original_roi = original_image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            # Determine the predominant color across different gamma corrections
            color = find_color(original_roi)
            # Apply OCR to the division
            psms_to_try = [6, 7, 8, 9, 10]
            text_blurred_task = asyncio.create_task(ocr_with_multiple_psms(thresh_roi, psms_to_try))
            text_blurred = await text_blurred_task

            if text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
                number = int(text_blurred)
            else:
                number = "Jolly"

            # Append the number and predominant color pair to the list
            current_run_numbers_and_colors.append([number,color])
            j += 1

        i += 1

    # Append current run numbers and colors to the tracking list
    numbers_at_positions.append(current_run_numbers_and_colors)
    #logger.info(current_run_numbers_and_colors)
# Function to find rectangular contours with a fixed threshold
async def find_rectangular_contours_with_fixed_threshold(image, target_num_rectangles):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numbers_at_positions = []  # List to keep track of numbers found at each position
    tasks = []
    thresh_max_value = 255
    continue_count = 0  # Counter to track the number of additional iterations
    for thresh_value in range(120, thresh_max_value, 1):
        if continue_count > 0:
            continue_count -= 1
            if continue_count == 0:
                break  # Stop the loop after 10 additional iterations
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        logger.info(f'Currently at : {thresh_value}')
        median = cv2.medianBlur(thresh, 7)
        contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        filtered_contours = [cnt for cnt in contours if
                             5000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0]) / 2]

        rectangular_contours = []
        heights = []
        for cnt in filtered_contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:  # Ensure the contour is rectangular
                x, y, w, h = cv2.boundingRect(approx)
                heights.append(h)
                rectangular_contours.append(cnt)
        rectangular_contours = group_contours_by_height(rectangular_contours, 200)

        # Sort rectangular contours based on area and select the top 10
        if len(rectangular_contours) != target_num_rectangles:
            continue  # Skip to the next threshold if the count doesn't match the target
        # Check if all heights are approximately the same
        height_threshold = 200  # You can adjust this threshold as needed
        if max(heights) - min(heights) >= height_threshold:
            logger.error("Rectangles do not have approximately the same height.")
            continue
        else:
            draw_rectangles(image, rectangular_contours)
            cv2.imshow("Image with rectangles", image)
            cv2.waitKey(0)
            continue_count = 1
            for thresh_adaptive_parameter in range(11, 121, 20):
                thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY, thresh_adaptive_parameter, 2)
                blurred_adaptive = cv2.GaussianBlur(thresh_adaptive, (3, 3), 0)
                median_adaptive = cv2.medianBlur(blurred_adaptive, 3)
                tasks.append(asyncio.create_task(
                    process_divisions_async(rectangular_contours, image, median_adaptive,
                                            numbers_at_positions, logger)))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    # Filter out current_run_numbers that are longer or shorter than the mode length
    if numbers_at_positions:
        lengths = [len(numbers) for numbers in numbers_at_positions]
        mode_length = Counter(lengths).most_common(1)[0][0]
        filtered_numbers_at_positions = [numbers for numbers in numbers_at_positions if len(numbers) == mode_length]

        # Determine the most predominant pair of number and color at each position
        positions_length = max(len(numbers) for numbers in filtered_numbers_at_positions)
        predominant_pairs = []

        for position in range(positions_length):
            pair_count_dict = {}

            for numbers in filtered_numbers_at_positions:
                if position < len(numbers):
                    pair = tuple(numbers[position])  # (number, color) as a tuple
                    if pair[0] != "Jolly":  # Exclude pairs with "Jolly" in the number section
                        if pair in pair_count_dict:
                            pair_count_dict[pair] += 1
                        else:
                            pair_count_dict[pair] = 1

            if pair_count_dict:
                predominant_pair = max(pair_count_dict, key=pair_count_dict.get)
                predominant_pairs.append(predominant_pair)
            else:
                predominant_pairs.append(("Jolly", None))  # Default pair if no valid pairs found

        return predominant_pairs



# Load the image
#merge pentru remiacsaV2_3,4 pt V2_5 vede Jolly in loc de 5 si 3 si remipebune cu height tolerance 100 px
image_path = 'MainApp/poze/4bd4a4cf-15e0-4822-ad18-aefb68a9d5f28763224859214459989.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (1980, 1080))
target_num_rectangles = 5# You can adjust this value

# Run the main function
loop = asyncio.get_event_loop()
result = loop.run_until_complete(find_rectangular_contours_with_fixed_threshold(image, target_num_rectangles))
logger.info(result)
