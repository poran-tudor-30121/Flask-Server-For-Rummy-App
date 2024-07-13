import cv2

# Global variables to store the coordinates of the ROI rectangle
start_x, start_y = -1, -1
end_x, end_y = -1, -1
drawing = False

# Mouse callback function to handle events
def draw_roi(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing

    # Left mouse button pressed: start drawing the rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    # Mouse movement: update the end coordinates of the rectangle if drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y

    # Left mouse button released: stop drawing the rectangle and calculate the area
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        # Draw the rectangle on the image
        cv2.rectangle(param, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Calculate the area of the selected region
        width = abs(end_x - start_x)
        height = abs(end_y - start_y)
        area = width * height
        print("Selected Area:", area)

        # Display the image with the selected rectangle
        cv2.imshow("Select ROI", param)

        # Uncomment this line if you want to close the window automatically after selection
        # cv2.waitKey(0)

# Load the image
image = cv2.imread('MainApp/poze/remiacasa5.jpg')

# Create a window and set the mouse callback function
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", draw_roi, param=image)

# Display the image
cv2.imshow("Select ROI", image)

# Wait for the user to select the ROI and press any key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
