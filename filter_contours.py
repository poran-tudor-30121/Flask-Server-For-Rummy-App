import cv2

def filter_contours(contours, min_area, max_area):
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    return filtered_contours
