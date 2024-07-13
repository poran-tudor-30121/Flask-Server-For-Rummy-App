import cv2
import numpy as np


def split_and_rotate_image(image_path):
    """
    Splits the image into four sections and rotates each section to standard orientation.

    :param image_path: str, Path to the input image.
    :return: dict, A dictionary containing the four sections as images.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")

    height, width = image.shape[:2]

    # Define sections
    sections = {
        'right': image[:, width // 2:],  # Right half of the image
        'left': image[:, :width // 2],  # Left half of the image
        'top': image[:height // 2, :],  # Top half of the image
        'bottom': image[height // 2:, :]  # Bottom half of the image
    }

    # Rotate sections
    rotated_sections = {
        'bottom': sections['bottom'],
        'right': rotate_image(sections['right'], 270),
        'left': rotate_image(sections['left'], 90),
        'top': rotate_image(sections['top'], 180),
    }

    return rotated_sections


def rotate_image(image, angle):
    """
    Rotates the given image by the specified angle.

    :param image: np.array, Image to be rotated.
    :param angle: float, Angle by which to rotate the image.
    :return: np.array, Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# Example usage
image_path = 'MainApp/poze/pozapejos3.jpg'
sections = split_and_rotate_image(image_path)

# Display the sections
for section_name, section_image in sections.items():
    cv2.imshow(section_name.capitalize(), section_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
