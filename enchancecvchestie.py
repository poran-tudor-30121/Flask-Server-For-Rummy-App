import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'MainApp/poze/remiacasa7.jpg'
image = cv2.imread(image_path)

# Define the ROI manually (assumed values; adjust as needed)
x, y, w, h = 150, 150, 500, 300  # Example values: (x, y, width, height)

# Create a mask for the ROI
mask = np.zeros_like(image)
mask[y:y+h, x:x+w] = 255

# Extract the ROI from the image
roi = image[y:y+h, x:x+w]

# Convert the ROI to grayscale
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to the ROI
equalized_roi = cv2.equalizeHist(gray_roi)

# Convert back to color
equalized_roi_color = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)

# Apply the enhanced ROI back to the original image
enhanced_image = image.copy()
enhanced_image[y:y+h, x:x+w] = equalized_roi_color

# Create a blurred version of the original image
blurred_image = cv2.GaussianBlur(image, (21, 21), 0)

# Combine the blurred background with the enhanced ROI
combined_image = np.where(mask==255, enhanced_image, blurred_image)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Enhanced Focus on Pieces')
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

# Save the result
result_path = 'MainApp/poze/enhanced_focus_image.jpg'
cv2.imwrite(result_path, combined_image)