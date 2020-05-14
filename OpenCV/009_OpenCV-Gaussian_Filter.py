import cv2
import numpy as np
from skimage import io

img = cv2.imread('../assets/imori_noise.jpg')

new = cv2.GaussianBlur(img, (3, 3), 1.3)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
