import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thre, new = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()