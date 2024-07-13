import cv2
import numpy as np

# Load the image
image = cv2.imread('MainApp/poze/remicluj.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 10, 100)

# Invert the edge image
edges_inverted = 255 - edges

# Display the inverted edge image
cv2.imshow('Inverted Edges', edges_inverted)
cv2.waitKey(0)

# Find contours in the inverted edge-detected image
contours, _ = cv2.findContours(edges_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# List to store rectangular contours
rectangles = []

# Iterate through each contour
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)

    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # Check if the contour has 4 corners (indicating a rectangle) and area is above a threshold
    if len(approx) == 4 and area > 100:
        rectangles.append(contour)

# Draw all contours on the original image in red
cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

# Draw the rectangular contours on the original image in green
cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)

# Display the image with all contours and rectangular contours
cv2.imshow('Contours and Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of rectangular contours found
print("Number of rectangular contours:", len(rectangles))
