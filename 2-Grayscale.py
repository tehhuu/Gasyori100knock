import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
new = new.astype(np.uint8)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()


