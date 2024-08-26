import cv2
import pytesseract
import numpy as np
from MainApp.identify_color import identify_color
from ocr_with_multiple_psms_first import ocr_with_multiple_psms_first
from multiple_players import classify_tiles

def custom_round(number, threshold=0.5):
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part >= threshold:
        return integer_part + 1
    else:
        return integer_part

# Load and preprocess the image
image = cv2.imread('MainApp/poze/remipebune.jpg')  # Adjust the path to your image file
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path to Tesseract executable

gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
equalized = cv2.equalizeHist(gray)

_, thresh = cv2.threshold(equalized, 215, 255, cv2.THRESH_BINARY)

median = cv2.medianBlur(thresh, 7)

cv2.imshow("Median",median)
cv2.waitKey(0)
blurred = cv2.GaussianBlur(thresh, (3, 3), 0)

contours, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
filtered_contours = [cnt for cnt in contours if 4000 < cv2.contourArea(cnt) < (image.shape[1] * image.shape[0]) / 2]

rectangular_contours = []
for contour in filtered_contours:
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        rectangular_contours.append(approx)

# Classify tiles based on orientation
tiles = classify_tiles(image, median, rectangular_contours)

# Find the smallest rectangular contour
smallest_contour = min(rectangular_contours, key=cv2.contourArea)
x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)

number_color_list = []

# Process each tile
for approx in rectangular_contours:
    x, y, w, h = cv2.boundingRect(approx)
    divisions = custom_round((w / w_smallest))
    if divisions < 1:
        divisions = 1

    for i in range(divisions):
        x_divided = int(x + (i * w_smallest))
        w_divided = int(w / divisions)
        upper_half_roi = blurred[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
        kernel = np.ones((3, 3), np.uint8)
        upper_half_roi = cv2.dilate(upper_half_roi, kernel)
        colored_roi = image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
        gray_roi = equalized[y:y + int(h // 1.5), x_divided:x_divided + w_divided]

        gray_roi_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        median = cv2.medianBlur(gray_roi_thresh, 3)
        blurred = cv2.GaussianBlur(median, (3, 3), 0)

        gray_roi_thresh = blurred[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
        gray_roi_thresh = cv2.resize(gray_roi_thresh, (300, 300))
        gray_roi = cv2.resize(gray_roi, (300, 300))
        upper_half_roi = cv2.resize(upper_half_roi, (300, 300))

        psms_to_try = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        text_blurred = ocr_with_multiple_psms_first(upper_half_roi, psms_to_try)
        text_gray = ocr_with_multiple_psms_first(gray_roi, psms_to_try)
        text_thresh = ocr_with_multiple_psms_first(gray_roi_thresh, psms_to_try)

        if text_blurred.strip().isdigit() and 1 <= int(text_blurred) <= 13:
            number = int(text_blurred)
        else:
            number = "Jolly"

        hsv_roi = cv2.cvtColor(image[y:y + int(h // 1.5), x_divided:x_divided + w_divided], cv2.COLOR_BGR2HSV)
        color = identify_color(hsv_roi)
        number_color_list.append([number, color])

        # Draw contours and detected numbers
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        cv2.putText(image, f'{number}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print(f'Tiles: {number_color_list}')
cv2.imshow('Identified Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
