import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

def binarization(num):
    if num > 128:
        return 255
    return 0

img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

# グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 2値化
_, new = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()