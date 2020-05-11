import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assets/imori_256x256_dark.png')

plt.hist(img.ravel(), bins=255, rwidth=1.0, range=(0, 255))
plt.show()
