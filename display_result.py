import cv2

def display_result(image, filtered_contours, number_color_list):
    for contour in filtered_contours:
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
    cv2.imshow('Identified Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
