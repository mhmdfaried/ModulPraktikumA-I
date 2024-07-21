import cv2
import numpy as np

# Open a connection to the webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the lower and upper bounds of the color to track
# These values can be adjusted to detect different colors
lower_color = np.array([66, 98, 100])  # Lower bound of the color in HSV
upper_color = np.array([156, 232, 255])  # Upper bound of the color in HSV

while True:
    # Capture a frame from the webcam
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to filter out the specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame, mask, and result
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Break the loop if the 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # ASCII value for 'Esc' key is 27
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
