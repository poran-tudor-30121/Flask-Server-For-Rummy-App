# Adjust saturation and brightness in the HSV image

import cv2
import pytesseract

# Load the image
image = cv2.imread('../MainApp/poze/remi.png')
#image = cv2.imread('Inchis-pe-tabla-la-remi-1024x454-1.jpg')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this with your Tesseract installation path


# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:, :, 1] = hsv[:, :, 1] * 1.05 # Increase saturation
hsv[:, :, 2] = hsv[:, :, 2] * 1# Increase brightness
cv2.imshow('HSV Image', hsv)
cv2.waitKey(0)
# Convert the enhanced HSV image back to BGR
enhanced_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# Display the enhanced color image
cv2.imshow('Enhanced Color Image', enhanced_color)
cv2.waitKey(0)