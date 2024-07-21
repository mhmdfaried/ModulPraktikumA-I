import cv2
import numpy as np

# 1. Konversi citra ke grayscale
img = cv2.imread('image_with_noise.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Salin citra asli
img_out = gray.copy()

# 3. Ukuran tinggi dan lebar citra
h, w = gray.shape

# 4. Penerapan filter max
for i in range(3, h - 3):
    for j in range(3, w - 3):
        neighbors = []
        for k in range(-3, 4):
            for l in range(-3, 4):
                neighbors.append(gray[i + k, j + l])
        max_val = max(neighbors)
        img_out.itemset((i, j), max_val)

# 5. Tampilkan citra hasil
cv2.imshow('Max Filter Result', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()