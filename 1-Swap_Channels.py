import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

img = cv2.imread('assets/imori.jpg')
#img = img[..., ::-1]
blue = img[:, :, 0].copy()
red = img[:, :, 2].copy()

new = img.copy()
new[:, :, 2], new[:, :, 0] = blue, red

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindow()

