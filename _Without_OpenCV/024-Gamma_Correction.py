import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate #list(accumulate(A))


def Gamma_Correction(img, c, g):
    img = img.astype(np.float)
    img /= 255

    new = pow(img / c, 1/g)

    new *= 255
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)

    return new


img = cv2.imread('../assets/imori_gamma.jpg')

new = Gamma_Correction(img, 1, 2.2)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
