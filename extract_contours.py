import cv2

def extract_contours(image):
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    return contours
