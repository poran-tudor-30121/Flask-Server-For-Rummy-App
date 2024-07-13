import cv2

# Load the image
image = cv2.imread('MainApp/poze/remiacasa.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to enhance contrast
equalized = cv2.equalizeHist(gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

# Apply non-local means denoising to remove salt and pepper noise
median = cv2.medianBlur(thresh, 5)

# Find contours
contours, _ = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and identify rectangles
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) == 4:
        cv2.drawContours(median, [approx], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Rectangles', median)
cv2.waitKey(0)
cv2.destroyAllWindows()
