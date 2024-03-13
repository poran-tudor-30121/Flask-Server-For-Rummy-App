import math

import cv2
import pytesseract
import numpy as np
from identify_color import identify_color
from scipy import ndimage

# Data viitoare - github repo , mai multe min contours
# Load the image
#image = cv2.imread('remi2.png')   # totu ok :)
#image = cv2.imread('Inchis-pe-tabla-la-remi-1024x454-1.jpg') #totu ok
#image = cv2.imread('remipebune.jpg') # totu ok!!!
#image = cv2.imread('remi.png') # totu ok :x
#image = cv2.imread('remi3.png') nu stiu exact ce se intampla.
#if image.shape[1] < 300 or image.shape[0] < 300:
    #image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

# image = ndimage.rotate(image, -1,reshape=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Update this with your Tesseract installation path

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
# Display the grayscale image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
# Aplicați egalizarea histogramei pentru a îmbunătăți contrastul
equalized = cv2.equalizeHist(gray)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 187, 255, cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the binary image
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)

median = cv2.medianBlur(thresh, 7)

cv2.imshow("Salt", median)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
# You can adjust the kernel size (e.g., (5, 5)) and standard deviation (e.g., 0)
cv2.imshow("Gaussian Blur", blurred)
cv2.waitKey(0)
# Find contours in the binary image
contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
print(f'Number of Contours: {len(contours)}')

# Iterate through contours and filter out small ones
filtered_contours = [cnt for cnt in contours if 4000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0])/2]
print({image.shape[1]*image.shape[0]},image.shape[1],image.shape[0])
# Ignore the last contour
#filtered_contours = filtered_contours[:-1]
# Find the smallest contour
smallest_contour = min(filtered_contours, key=cv2.contourArea)
# Calculate the bounding box of the smallest contour
x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
print(f'Smalles values:{x_smallest, y_smallest, w_smallest, h_smallest}')

# print(f'Number of Filtered Contours: {len(filtered_contours)}')

number_color_list = []  # List to store pairs of number and color
# Iterate through filtered contours
for contour in filtered_contours:
    # Approximate a polygon (rectangle) around the contour
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the contour is rectangular
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        # Calculate the number of divisions based on the smallest contour size
        divisions = round((w / w_smallest))
        print(f'Number of divisions: {divisions}')
        if divisions < 1:
            divisions = 1
        # Extract upper half of the bounding box

        for i in range(divisions):
            x_divided = int(x + (i * w_smallest))
            w_divided = int(w / divisions)
            upper_half_roi = blurred[y:y + int(h // 1.5), x_divided:x_divided + w_divided]

            kernel = np.ones((3, 3), np.uint8)
            upper_half_roi = cv2.dilate(upper_half_roi, kernel)

            colored_roi = image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
            print(f'Uperr_half_roi_values:{x_divided, y, w_divided, h}')
            # Apply OCR on the ROI
            text = pytesseract.image_to_string(upper_half_roi, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            print(f'Initial text {text}')
            # Check if OCR result is a numerical value between 1 and 13
            if text.strip().isdigit() and 1 <= int(text) <= 13:
                number = int(text)
            else:
                number = "Jolly"  # Set the number to "Jolly" for non-numeric or out-of-range values
            # Color identification
            hsv_roi = cv2.cvtColor(image[y:y + int(h // 1.5), x_divided:x_divided + w_divided], cv2.COLOR_BGR2HSV)
            color = identify_color(hsv_roi)
            number_color_list.append([number, color])  # Add the pair to the list

            print(f'Numbers in Rectangle: Number {number}, Color: {color}')
            cv2.imshow('Upper Half,colored', upper_half_roi)
            cv2.waitKey(0)
            # Draw rectangles around the identified rectangles on the original image
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Display the final result with rectangles around identified numbers
print(f'Tiles: {number_color_list}')
cv2.imshow('Identified Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
