import cv2

def preprocess_image(image):
    #cv2.imshow('Original Image', image)
   # cv2.waitKey(0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # Display the grayscale image
   # cv2.imshow('Grayscale Image', gray)
    #cv2.waitKey(0)
    # Aplicați egalizarea histogramei pentru a îmbunătăți contrastul
    #equalized = cv2.equalizeHist(gray)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 197, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Display the binary image
   # cv2.imshow('Binary Image', thresh)
   # cv2.waitKey(0)

    return thresh
