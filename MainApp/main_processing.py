import asyncio
import logging

import numpy as np
from collections import Counter

import cv2
from identify_color import identify_color
from ocr_with_multiple_psms import ocr_with_multiple_psms
from utilities import group_contours_by_height, draw_rectangles,split_and_rotate_image
from init_config import initialize_logging


def find_color(original_roi):
    hsv_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2HSV)
    return identify_color(hsv_roi)


async def process_divisions_async(rectangular_contours, original_image, median_adaptive, numbers_at_positions, logger):
    current_run_numbers_and_colors = []
    smallest_contour = min(rectangular_contours, key=cv2.contourArea)
    x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
    w_smallest_adjusted = w_smallest // 3  # Adjust the width to be 1/3 of the smallest rectangular contour's width
    i = 0

    while i < len(rectangular_contours):
        approx = rectangular_contours[i]
        x, y, w, h = cv2.boundingRect(approx)
        divisions = round(w / w_smallest_adjusted)
        if divisions < 1:
            divisions = 1

        j = 0
        while j < divisions:
            x_divided = int(x + (j * w_smallest_adjusted))
            w_divided = int(w / divisions)

            thresh_roi = median_adaptive[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            thresh_roi = cv2.resize(thresh_roi, (300, 300))
            kernel = np.ones((3, 3), np.uint8)
            thresh_roi = cv2.dilate(thresh_roi, kernel)
            #cv2.imshow("Roi",thresh_roi)
            #cv2.waitKey(0)

            original_roi = original_image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            color = find_color(original_roi)

            psms_to_try = [7, 8, 9, 10, 11]
            text_blurred_task = asyncio.create_task(ocr_with_multiple_psms(thresh_roi, psms_to_try))
            text_blurred = await text_blurred_task

            if text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
                number = int(text_blurred)
            else:
                number = "Jolly"

            current_run_numbers_and_colors.append([number, color])
            j += 1

        i += 1

    numbers_at_positions.append(current_run_numbers_and_colors)
    #logger.info(current_run_numbers_and_colors)


async def find_rectangular_contours_with_fixed_threshold(image, logger):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numbers_at_positions = []
    tasks = []
    thresh_max_value = 255
    continue_count = 0
    max_number_of_rectangles = 0
    formations = []
    for thresh_value in range(100, thresh_max_value, 1):
        if continue_count > 0:
            continue_count -= 1
            if continue_count == 0:
                break
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        logger.info(f'Currently at : {thresh_value}')
        median = cv2.medianBlur(thresh, 7)
        #cv2.imshow("median", median)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
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
                if w >= 2 * h:  # Ensure the width is at least double the height
                    heights.append(h)
                    rectangular_contours.append(cnt)

        rectangular_contours = group_contours_by_height(rectangular_contours, 20)
        if max_number_of_rectangles <= len(rectangular_contours):
            max_number_of_rectangles = len(rectangular_contours)
            formations = rectangular_contours

        # Sort rectangular contours based on area and select the top 10
        #if len(rectangular_contours) != target_num_rectangles:
            #continue  # Skip to the next threshold if the count doesn't match the target
        # Check if all heights are approximately the same
        #height_threshold = 20  # You can adjust this threshold as needed
        #if max(heights) - min(heights) >= height_threshold:
            #logger.error("Rectangles do not have approximately the same height.")
            #continue
       # else:
    if len(formations) <= 1:
        return []
    #draw_rectangles(image, formations)
    #cv2.imshow("Image with rectangles", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    continue_count = 1
    for thresh_adaptive_parameter in range(31, 121, 10):
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_adaptive_parameter, 2)
        blurred_adaptive = cv2.GaussianBlur(thresh_adaptive, (3, 3), 0)
        median_adaptive = cv2.medianBlur(blurred_adaptive, 3)
        tasks.append(asyncio.create_task(process_divisions_async(formations, image, median_adaptive, numbers_at_positions, logger)))

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
                predominant_pairs.append(("Jolly", "Black"))  # Default pair if no valid pairs found

        return predominant_pairs


async def process_image_for_all_players(image_path, logger):

    image = cv2.imread(image_path)
    # image = cv2.resize(image, (1980, 1080))
    sections = split_and_rotate_image(image)

    i = 0

    tasks = []
    for section_name, section_image in sections.items():
        tasks.append(asyncio.create_task(find_rectangular_contours_with_fixed_threshold(section_image, logger)))

    # Wait for all tasks to complete concurrently
    results_list = await asyncio.gather(*tasks)

    return results_list

async def process_image_for_some_players(image_path, logger, wanted_sections):
    image = cv2.imread(image_path)
    sections = split_and_rotate_image(image)

    tasks = []
    results_map = {section_name: None for section_name in wanted_sections}  # Initialize results_map with wanted_sections

    for section_name in wanted_sections:
        if section_name in sections:
            section_image = sections[section_name]
            task = asyncio.create_task(find_rectangular_contours_with_fixed_threshold(section_image, logger))
            tasks.append((section_name, task))
        else:
            logger.warning(f"Section {section_name} not found in the image sections")

    # Wait for all tasks to complete concurrently
    results_list = await asyncio.gather(*(task for _, task in tasks))

    # Store results in results_map based on section names
    for (section_name, _), result in zip(tasks, results_list):
        results_map[section_name] = result

    # Convert results_map to list in the order of wanted_sections
    ordered_results = [results_map[section_name] for section_name in wanted_sections if section_name in results_map]

    return ordered_results

