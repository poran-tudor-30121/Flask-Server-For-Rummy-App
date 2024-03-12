import cv2
import pytesseract
import numpy as np
from identify_color import identify_color
from ocr_with_multiple_psms import ocr_with_multiple_psms

def identify_numbers_colors(blurred,image, filtered_contours, w_smallest):
    number_color_list = []
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    for contour in filtered_contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            divisions = round((w / w_smallest))
            if divisions < 1:
                divisions = 1
            for i in range(divisions):
                x_divided = int(x + (i * w_smallest))
                w_divided = int(w / divisions)
                upper_half_roi = blurred[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
                kernel = np.ones((3, 3), np.uint8)
                upper_half_roi = cv2.dilate(upper_half_roi, kernel)
                colored_roi = image[y:y + int(h // 1.5), x_divided:x_divided + w_divided]
                psms_to_try = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                number = ocr_with_multiple_psms(upper_half_roi, psms_to_try)
                hsv_roi = cv2.cvtColor(image[y:y + int(h // 1.5), x_divided:x_divided + w_divided], cv2.COLOR_BGR2HSV)
                color = identify_color(hsv_roi)
                number_color_list.append([number, color])
                print(f'Numbers in Rectangle: Number {number}, Color: {color}')
                cv2.imshow('Upper Half', upper_half_roi)
                cv2.waitKey(0)
    return number_color_list

