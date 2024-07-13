import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
image = cv2.imread('MainApp/poze/remiacasaV2_2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

async def find_rectangular_contours_with_otsu(image, target_num_rectangles):
    if target_num_rectangles == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #equalized = cv2.equalizeHist(gray)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    #cl1 = clahe.apply(equalized)
    numbers_at_positions = []  # List to keep track of numbers found at each position
    tasks = []

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    ret, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(otsu_thresh, 7)
    cv2.imshow("Median_Otsu", median)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    filtered_contours = [cnt for cnt in contours if 5000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0]) / 2]
    rectangular_contours = []
    heights = []
    for cnt in filtered_contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:  # Ensure the contour is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            heights.append(h)
            rectangular_contours.append(cnt)

    # Sort rectangular contours based on area and select the top 10
    if len(rectangular_contours) <= target_num_rectangles:
        return None  # Return None if the count doesn't match the target

    # Check if all heights are approximately the same
    height_threshold = 200  # You can adjust this threshold as needed
    if max(heights) - min(heights) >= height_threshold:
        return None  # Return None if rectangles do not have approximately the same height



    for thresh_adaptive_parameter in range(51, 121, 20):
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
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

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's thresholding
ret, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the result
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu's Thresholding")
plt.axis('off')

plt.show()
