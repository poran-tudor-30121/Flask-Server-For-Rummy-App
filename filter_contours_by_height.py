import cv2


def filter_contours_by_height(contours, smallest_contour_height, height_threshold):
    filtered_contours = []
    for contour in contours:
        _, _, _, contour_height = cv2.boundingRect(contour)
        print(contour_height,smallest_contour_height)
        # Compare the contour height to the threshold
        if contour_height < smallest_contour_height * height_threshold:
            filtered_contours.append(contour)
    return filtered_contours
