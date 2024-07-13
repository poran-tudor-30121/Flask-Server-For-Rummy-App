import cv2
import numpy as np

# Load the image
image = cv2.imread('MainApp/poze/remicluj.jpg')

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Display the blurred image
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(0)

# Apply adaptive thresholding to extract white regions
thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)

median = cv2.medianBlur(thresholded, 5)
cv2.imshow('Salt Pepper', median)
cv2.waitKey(0)

# Find contours in the noisy image
contours, _ = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store rectangular bounding boxes
rectangular_boxes = []

# Iterate through each contour
for contour in contours:
    # Calculate the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Ensure the bounding box is sufficiently large to avoid noise
    if w > 10 and h > 10:
        rectangular_boxes.append((x, y, w, h))

# Draw the rectangular bounding boxes on the original image
result_image = np.copy(image)
for (x, y, w, h) in rectangular_boxes:
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result image with rectangular bounding boxes
cv2.imshow('Rectangular Tiles', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of rectangular bounding boxes found
print("Number of rectangular tiles:", len(rectangular_boxes))
