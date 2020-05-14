import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


