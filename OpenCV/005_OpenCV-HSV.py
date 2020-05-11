import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math


img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

# HSV変換
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# Hueの値を反転
img_hsv[0, ...] = (img_hsv[0, ...] + 90) % 180
# RGBに戻す
new = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()