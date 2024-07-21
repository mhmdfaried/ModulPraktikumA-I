import cv2
import numpy as np

# Load the first image (img1.jpg)
img1 = cv2.imread('../koala.jpeg', 0)

# Load the second image (img2.jpg)
img2 = cv2.imread('../koala2.jpg', 0)

# Check if both images are loaded successfully
if img1 is None or img2 is None:
    print("One or more images could not be loaded.")
else:
    # Perform the addition operation
    add_img = cv2.add(img1, img2)

    # Perform the subtraction operation (img1 - img2)
    subtract_img = cv2.subtract(img1, img2)

    # Display the result or do further processing
    cv2.imshow("Added Image", add_img)
    cv2.imshow("Subtracted Image", subtract_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()