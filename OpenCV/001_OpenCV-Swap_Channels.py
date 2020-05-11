import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

img = cv2.imread('assets/imori.jpg')

new = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()


