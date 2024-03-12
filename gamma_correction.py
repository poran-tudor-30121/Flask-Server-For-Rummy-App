import cv2
import numpy as np
def gamma_correction(image, gamma=1.0):
    # Apply gamma correction to the image
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected