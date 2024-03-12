import cv2
import pytesseract
import numpy as np
from preprocess_image import preprocess_image
from extract_contours import extract_contours
from filter_contours import filter_contours
from identify_numbers_colors import identify_numbers_colors
from display_result import display_result
from gamma_correction import gamma_correction
from apply_blurs import apply_blurs
from filter_contours_by_height import filter_contours_by_height
from scipy import ndimage

def main():
    image = cv2.imread('remi3.png')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    median, blurred = apply_blurs(preprocessed_image)

    # Extract contours
    contours = extract_contours(median)
    # Filter contours
    filtered_contours = filter_contours(contours, (image.shape[1] * image.shape[0]) / 150,
                                        (image.shape[1] * image.shape[0]) / 4)
    if len(filtered_contours) < 2:
     gamma_corrected_image = gamma_correction(image, gamma=0.5)
     #image = gamma_correction(image, gamma=1.2)
     preprocessed_image_gamma_corrected = preprocess_image(gamma_corrected_image)
     cv2.imshow('Gammma correction', preprocessed_image_gamma_corrected)
     cv2.waitKey(0)

     median, blurred = apply_blurs(preprocessed_image_gamma_corrected)

     # Extract contours
     contours = extract_contours(median)
     # Filter contours
     filtered_contours = filter_contours(contours, (image.shape[1] * image.shape[0]) / 50,
                                         (image.shape[1] * image.shape[0]) / 2)
    # Find the smallest contour
    smallest_contour = min(filtered_contours, key=cv2.contourArea)
    x_smallest, y_smallest, w_smallest, h_smallest = cv2.boundingRect(smallest_contour)
    filtered_contours=filter_contours_by_height(filtered_contours, h_smallest, 1.3)

    # Identify numbers and colors within contours
    number_color_list = identify_numbers_colors(blurred, image, filtered_contours, w_smallest)
    print("Number-Color List:")
    for item in number_color_list:
        print(item)
    # Display the final result
    display_result(image, filtered_contours, number_color_list)


if __name__ == "__main__":
    main()
