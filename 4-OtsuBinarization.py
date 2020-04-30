import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new


def Otsubinarization(img):

    def binarization(num, thre):
        if num > thre:
            return 255
        return 0
    
    best_thre = max_dbc = -1
    for thre in range(1, 256):
        w_1 = np.count_nonzero(img >= thre)
        w_2 = np.count_nonzero(img < thre)
        m_1 = img[img >= thre].mean()
        m_2 = img[img < thre].mean()
        if math.isnan(m_1): m_1 = 0
        if math.isnan(m_2): m_2 = 0
        dbc = (w_1 * w_2 / (w_1+w_2)**2) * ((m_1-m_2)**2)
        if dbc > max_dbc:
            best_thre = thre
            max_dbc = dbc

    new = np.vectorize(binarization)(img, best_thre)
    new = new.astype(np.uint8)
    return new


img = cv2.imread('assets/imori.jpg')
img = img[..., ::-1]

new = Grayscale(img) #グレースケール化
new = Otsubinarization(new) #大津の2値化

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()