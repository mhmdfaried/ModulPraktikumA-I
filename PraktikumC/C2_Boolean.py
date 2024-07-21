import cv2
import numpy as np

# Load the first image (img1.jpg)
img1 = cv2.imread('../koala.jpeg', 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# Load the second image (img2.jpg)
img2 = cv2.imread('../koala2.jpg', 1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Check if both images are loaded successfully
# Perform bitwise AND operation
result = cv2.bitwise_and(img1, img2)

# Display the result or do further processing
cv2.imshow("Bitwise AND Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()