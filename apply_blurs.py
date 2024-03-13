import cv2


def apply_blurs(image):
    # Apply median blur
    median = cv2.medianBlur(image, 7)
    #cv2.imshow('Salt Pepper', median)
    #cv2.waitKey(0)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    #cv2.imshow('Blurred', blurred)
    #cv2.waitKey(0)

    return median, blurred
