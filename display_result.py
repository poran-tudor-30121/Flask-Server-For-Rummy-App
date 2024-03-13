import cv2

def display_result(image, filtered_contours, number_color_list):
    for contour in filtered_contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
    return image