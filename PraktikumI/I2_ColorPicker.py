import cv2
import numpy as np

def nothing(x):
    pass

# Open a connection to the webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window named "Trackbars" for the trackbars
cv2.namedWindow("Trackbars")

# Create trackbars for adjusting the HSV color space
cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    # Capture a frame from the webcam
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current positions of the trackbars
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    # Define the lower and upper bounds of the color to track
    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

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
