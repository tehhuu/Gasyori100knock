import cv2
import numpy as np


img = cv2.imread('../assets/imori_noise.jpg')

new = cv2.medianBlur(img, 3)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
