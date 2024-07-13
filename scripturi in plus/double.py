import math

import cv2
import pytesseract
import numpy as np

# Load the image
image = cv2.imread('../MainApp/poze/remi.png')
#image = cv2.imread('Inchis-pe-tabla-la-remi-1024x454-1.jpg')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this with your Tesseract installation path

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (3, 3),
                          0)
# You can adjust the kernel size (e.g., (5, 5)) and standard deviation (e.g., 0)
cv2.imshow("Gaussian Blur", blurred)
cv2.waitKey(0)


# Apply thresholding to create a binary image
_, thresh = cv2.threshold(blurred, 187, 255, cv2.THRESH_BINARY)

# Display the binary image
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
print(f'Number of Contours: {len(contours)}')

# Iterate through contours and filter out small ones
filtered_contours = [cnt for cnt in contours if 1000 < cv2.contourArea(cnt)]
# Ignore the last contour
filtered_contours = filtered_contours[:-1]
# Find the smallest contour
smallest_contour = min(filtered_contours, key=cv2.contourArea)
# Calculate the bounding box of the smallest contour
x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
print(f'Smalles values:{x_smallest, y_smallest, w_smallest, h_smallest}')

#print(f'Number of Filtered Contours: {len(filtered_contours)}')
zoom_factor = 1.1;
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
            w_divided = int(w/divisions)
            upper_half_roi = thresh[y:y + int(h//1.5), x_divided:x_divided + w_divided]
            print(f'Uperr_half_roi_values:{x_divided, y, w_divided, h}')
            # Apply OCR on the ROI
            text = pytesseract.image_to_string(upper_half_roi, config='--psm 6 ')
            #updated_text = 'no update needed';


            cv2.imshow('Upper Half', upper_half_roi)
            print(f'Numbers in Rectangle: {text}')
            cv2.waitKey(0)


        # Draw rectangles around the identified rectangles on the original image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Display the final result with rectangles around identified numbers
cv2.imshow('Identified Numbers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
