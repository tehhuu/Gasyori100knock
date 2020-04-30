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
new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
# 2値化
new2 = np.vectorize(binarization)(new)
new2 = new2.astype(np.uint8)

cv2.imshow('', new2)
cv2.waitKey(0)
cv2.destroyAllWindows()