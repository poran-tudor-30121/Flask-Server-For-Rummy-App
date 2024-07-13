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

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

# Draw all contours on the original image in red
contours_image = np.copy(image)
cv2.drawContours(contours_image, contours, -1, (0, 0, 255), 2)

# Display the image with edges and all contours
cv2.imshow('Edges and All Contours', contours_image)
cv2.waitKey(0)

# List to store rectangular contours
rectangles = []

# Iterate through each contour
for contour in contours:
    # Calculate the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate aspect ratio
    aspect_ratio = float(w) / h

    # Calculate area of the contour
    area = cv2.contourArea(contour)

    # Calculate arc length
    arc_length = cv2.arcLength(contour, True)

    # Check if contour properties resemble a rectangle
    if aspect_ratio >= 0.5 and aspect_ratio <= 2.0 and abs(arc_length - 2 * (w + h)) < 0.05 * arc_length:
        rectangles.append(contour)

# Draw the rectangles contours on the original image in green
cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)

# Display the image with rectangular contours
cv2.imshow('Rectangles Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of rectangles contours found
print("Number of rectangles contours:", len(rectangles))
