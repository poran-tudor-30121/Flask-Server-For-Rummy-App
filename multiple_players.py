import cv2
import numpy as np


def classify_tiles(original,image, contours):
    """
    Classify tiles based on their orientation into four players and label with red text.

    Args:
    - image: The processed image in which tiles are detected.
    - contours: List of contours representing the detected tiles.

    Returns:
    - classified_tiles: Dictionary with player names as keys and list of classified contours as values.
    """
    # Initialize dictionary to store classified tiles
    classified_tiles = {'Player 1': [], 'Player 2': [], 'Player 3': [], 'Player 4': []}

    # Initialize list to store contour data
    contour_data = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[2]

        # Correct angle calculation for bounding box
        if rect[1][0] < rect[1][1]:
            angle = 90 + angle

        # Store contour and angle information
        contour_data.append((contour, rect, angle))

    # Determine the smallest rectangular contour for reference
    smallest_contour_data = min(contour_data, key=lambda x: cv2.contourArea(x[0]))
    _, smallest_rect, _ = smallest_contour_data

    # Classify tiles based on orientation
    for contour, rect, angle in contour_data:
        tile_center = rect[0]
        angle_difference = angle - smallest_rect[2]

        # Normalize angle to be between 0 and 180
        normalized_angle = angle_difference % 180

        # Classify based on angle and expected player orientation
        if 75 <= normalized_angle <= 105:
            player = 'Player 2'  # Horizontal to the right
            contour_color = (255, 0, 0)  # Blue
        elif 165 <= normalized_angle or normalized_angle <= 15:
            player = 'Player 4'  # Horizontal to the left
            contour_color = (0, 255, 255)  # Yellow
        elif 30 <= normalized_angle <= 60:
            player = 'Player 1'  # Vertical with numbers downwards
            contour_color = (0, 255, 0)  # Green
        else:
            player = 'Player 3'  # Vertical with numbers upside down
            contour_color = (0, 0, 255)  # Red

        # Store the classified contour
        classified_tiles[player].append(contour)

        # Draw contours
        cv2.drawContours(image, [np.int0(cv2.boxPoints(rect))], -1, contour_color, 2)

        # Draw red text labels for players
        text_color = (0, 0, 255)  # Red
        cv2.putText(original, player, (int(tile_center[0]), int(tile_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Show the classified image for verification
    cv2.imshow('Classified Tiles', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return classified_tiles

# Example usage (commented out as it's for illustration):
# processed_image = cv2.imread('/mnt/data/pozajucatorimultiplii.jpg')
# contours = [list of contours obtained from contour detection]
# classified_tiles = classify_tiles(processed_image, contours)
# print(classified_tiles)
