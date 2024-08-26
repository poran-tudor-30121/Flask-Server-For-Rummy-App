import cv2
import numpy as np
from MainApp.identify_color import identify_color


def draw_rectangles(image, rectangles):
    for rect in rectangles:
        x, y, w, h =  cv2.boundingRect(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_contours(image, contours):
    # Make a copy of the input image to draw on
    output_image = image.copy()

    # Draw the contours on the image
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    return output_image


def get_contour_height(contour):
    _, _, _, h = cv2.boundingRect(contour)
    return h


def get_contour_area(contour):
    return cv2.contourArea(contour)


def group_contours_by_height(contours, tolerance):
    if not contours:
        return []

    contours_sorted = sorted(contours, key=get_contour_height)
    grouped_contours = []
    current_group = []

    for contour in contours_sorted:
        if not current_group or abs(get_contour_height(contour) - get_contour_height(current_group[0])) <= tolerance:
            current_group.append(contour)
        else:
            grouped_contours.append(current_group)
            current_group = [contour]

    if current_group:
        grouped_contours.append(current_group)

    if not grouped_contours:
        return []

    # Find the group with the maximum number of rectangles
    max_size = max(len(group) for group in grouped_contours)
    max_size_groups = [group for group in grouped_contours if len(group) == max_size]

    # If there's a tie, choose the group with the largest area overall
    if len(max_size_groups) > 1:
        max_size_groups.sort(key=lambda group: sum(get_contour_area(contour) for contour in group), reverse=True)

    # Return the group with the largest number of rectangles and largest area in case of a tie
    return max_size_groups[0] if max_size_groups else []
def rotate_image(image, angle):
    # Get the image dimensions
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Compute the bounding box of the new image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated


def split_and_rotate_image(image):
    height, width = image.shape[:2]

    sections = {
        'bottom': image[height // 2:, :],
        'top': rotate_image(image[:height // 2, :], 180),
        'left': rotate_image(image[:, :width // 2], 90),
        'right': rotate_image(image[:, width // 2:], 270)
    }

    return sections
def show_sections(sections):
    for name, section in sections.items():
        cv2.imshow(name, section)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_color(original_roi):
    hsv_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2HSV)
    color = identify_color(hsv_roi)
    return color