import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

def ColorReduction(img):
    new = np.zeros_like(img, dtype=np.uint8)
    val_list = [32, 96, 160, 224]
    for i in range(3):
        for j in range(4):
            ind = np.where((64*j <= img[..., i]) & (img[..., i] < 64*(j+1)))
            new[..., i][ind] = val_list[j]
    return new
            

img = cv2.imread('assets/imori.jpg')

new = ColorReduction(img)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()