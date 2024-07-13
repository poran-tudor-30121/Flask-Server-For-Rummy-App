import cv2
import numpy as np
from identify_numbers_colors import identify_numbers_colors

# Load the image
image = cv2.imread('MainApp/poze/remiacasa5.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)



# Apply Canny edge detection
edges = cv2.Canny(blurred, 10, 100)

cv2.imshow('Edges', edges)
cv2.waitKey(0)


# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=200, param2=30, minRadius=5, maxRadius=20)

# If circles are detected
if circles is not None:
    # Convert the coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # Draw the circles on the original image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Change the color and thickness as needed
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Draw the center of the circle

    # Display the image with circles
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    print(len(circles))
    cv2.destroyAllWindows()
else:
    print("No circles detected.")


# Filter only rectangles among the detected contours
rectangles = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.08
                              * perimeter, True)
    if len(approx) == 4:
        rectangles.append(contour)
print(len(rectangles))
for rectangle in rectangles:
    print (cv2.contourArea(rectangle))
# Find the smallest rectangle among the detected rectangles with an area larger than 100
min_area_threshold = 20000
for rectangle in rectangles:
    if cv2.contourArea(rectangle) > min_area_threshold:
        smallest_rectangle = rectangle
        break
else:
    # If no rectangle with area larger than 100 is found, exit
    print("No rectangle with area larger than 100 found")
    cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)
    cv2.imshow('Filtered Contours 2', image)
    cv2.waitKey(0)
    exit()

# Get the height of the smallest rectangle
cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)
cv2.imshow('Filtered Contours 3', image)
cv2.waitKey(0)
x, y, w, h = cv2.boundingRect(smallest_rectangle)
min_rectangle_height = h

# Filter contours based on the height of the smallest rectangle and the height of other contours
filtered_contours = [contour for rectangle in rectangles if
                     min_rectangle_height * 1.2 > cv2.boundingRect(contour)[3] > min_rectangle_height * 0.8]

# Draw the filtered contours on the original image
cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

# Identify numbers and colors within contours
#number_color_list = identify_numbers_colors(blurred, image, filtered_contours, w)
#print("Number-Color List:")
#for item in number_color_list:
    #print(item)
# Display the final image with filtered contours
cv2.imshow('Filtered Contours 4', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
