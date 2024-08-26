import asyncio

import numpy as np
from collections import Counter

import cv2
from MainApp.identify_color import identify_color
from MainApp.ocr_with_multiple_psms import ocr_with_multiple_psms
from utilities import group_contours_by_height, draw_rectangles, split_and_rotate_image, draw_contours


def find_color(original_roi):
    hsv_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2HSV)
    return identify_color(hsv_roi)


def process_divisions_sync(rectangular_contours, original_image, median_adaptive, numbers_at_positions, logger):
    current_run_numbers_and_colors = []
    smallest_contour = min(rectangular_contours, key=cv2.contourArea)
    x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
    w_smallest_adjusted = w_smallest // 3  # Adjust the width to be 1/3 of the smallest rectangular contour's width

    for approx in rectangular_contours:
        x, y, w, h = cv2.boundingRect(approx)
        divisions = round(w / w_smallest_adjusted)
        if divisions < 1:
            divisions = 1

        for j in range(divisions):
            x_divided = int(x + (j * w_smallest_adjusted))
            w_divided = int(w / divisions)

            thresh_roi = median_adaptive[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            thresh_roi = cv2.resize(thresh_roi, (300, 300))
           # cv2.imshow("Thresh ROI Adaptive", thresh_roi)
            #cv2.waitKey(0)
          #  cv2.destroyAllWindows()
            kernel = np.ones((3, 3), np.uint8)
            thresh_roi = cv2.dilate(thresh_roi, kernel)
            i = 0
            cv2.imshow("Thresh ROI Adaptive After Kernel", thresh_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i+=1

            original_roi = original_image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            color = find_color(original_roi)

            psms_to_try = [7, 8, 9, 10, 11]
            text_blurred = ocr_with_multiple_psms(thresh_roi, psms_to_try)

            if text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
                number = int(text_blurred)
            else:
                number = "Jolly"

            current_run_numbers_and_colors.append([number, color])

    numbers_at_positions.append(current_run_numbers_and_colors)

def find_rectangular_contours_with_fixed_threshold(image, logger):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    numbers_at_positions = []
    thresh_max_value = 255
    max_number_of_rectangles = 0
    formations = []

    for thresh_value in range(100, thresh_max_value, 1):
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Binary Image", thresh)
       # cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #logger.info(f'Currently at : {thresh_value}')
        median = cv2.medianBlur(thresh, 7)
        #cv2.imshow("Binary Image Median Blurred", thresh)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        filtered_contours = [cnt for cnt in contours if 5000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0]) / 2]

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
            contoursUnf = contours
            filtered_contoursUnf = filtered_contours

    if len(formations) <= 1:
        return []

    contours_img = draw_contours(image,contoursUnf)
   # cv2.imshow("Contours", contours_img)
   # cv2.waitKey(0)
  #  cv2.destroyAllWindows()

    filtered_contours_img = draw_contours(image,filtered_contoursUnf)
    #cv2.imshow("Filtered contours", filtered_contours_img)
    #cv2.waitKey(0)
   # cv2.destroyAllWindows()

    draw_rectangles(image, formations)
    #cv2.imshow("Image with rectangles", image)
    #cv2.waitKey(0)
   # cv2.destroyAllWindows()

    with ThreadPoolExecutor() as executor:
        futures = []
        for thresh_adaptive_parameter in range(31, 121, 10):
            thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_adaptive_parameter, 2)
           # cv2.imshow("Thresh Adaptive {thresh_adaptive_parameter}",thresh_adaptive)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            blurred_adaptive = cv2.GaussianBlur(thresh_adaptive, (3, 3), 0)
           # cv2.imshow("Blurred Adaptive {thresh_adaptive_parameter}", blurred_adaptive)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            median_adaptive = cv2.medianBlur(blurred_adaptive, 3)
           # cv2.imshow("Median Adaptive {thresh_adaptive_parameter}", median_adaptive)
            #cv2.waitKey(0)
           # cv2.destroyAllWindows()
            futures.append(executor.submit(process_divisions_sync, formations, image, median_adaptive, numbers_at_positions, logger))

        for future in futures:
            future.result()

    if numbers_at_positions:
        lengths = [len(numbers) for numbers in numbers_at_positions]
        mode_length = Counter(lengths).most_common(1)[0][0]
        filtered_numbers_at_positions = [numbers for numbers in numbers_at_positions if len(numbers) == mode_length]

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
    #image = cv2.resize(image, (1980, 1080))
    sections = split_and_rotate_image(image)

    i = 0

    tasks = []
    for section_name, section_image in sections.items():
        tasks.append(asyncio.create_task(find_rectangular_contours_with_fixed_threshold(section_image, logger)))
        cv2.imshow("section",section_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Wait for all tasks to complete concurrently
    results_list = await asyncio.gather(*tasks)

    return results_list

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def process_image_section(section_name, section_image, logger):
    return section_name, find_rectangular_contours_with_fixed_threshold(section_image, logger)

async def process_image_for_some_players(image_path, logger, wanted_sections):
    image = cv2.imread(image_path)
    sections = split_and_rotate_image(image)
    image1 = cv2.resize(image, (1600, 1080))
    sections1 = split_and_rotate_image(image1)
    #show_sections(sections1)

    results_map = {section_name: None for section_name in wanted_sections}

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        futures = {}
        for section_name in wanted_sections:
            if section_name in sections:
                section_image = sections[section_name]
                future = loop.run_in_executor(executor, process_image_section, section_name, section_image, logger)
                futures[section_name] = future
            else:
                logger.warning(f"Section {section_name} not found in the image sections")

        for section_name, future in futures.items():
            result = await future
            results_map[section_name] = result

    ordered_results = [results_map[section_name] for section_name in wanted_sections if section_name in results_map]

    return ordered_results

