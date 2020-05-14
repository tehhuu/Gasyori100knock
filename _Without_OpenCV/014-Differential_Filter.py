import cv2
import numpy as np

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new


def h_Differential_Filter(img):
    H, W = img.shape
    _img = img.copy().astype(np.float)
    new = _img.copy()

    for i in range(H):
        for j in range(W-1):
            new[i, j] = _img[i, j+1] - _img[i, j]
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new

def v_Differential_Filter(img):
    H, W = img.shape
    _img = img.copy().astype(np.float)
    new = _img.copy()

    for i in range(H-1):
        for j in range(W):
            new[i, j] = _img[i+1, j] - _img[i, j]
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new


def synthesize_grayscale(img1, img2):
    _img1 = img1.copy().astype(np.float)
    _img2 = img2.copy().astype(np.float)
    new = np.sqrt(_img1**2 + _img2**2)
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new


img = cv2.imread('../assets/imori.jpg')

gray = Grayscale(img)
new_h = h_Differential_Filter(gray)
new_v = v_Differential_Filter(gray)
#横方向と縦方向を合成してみた
new_mul = synthesize_grayscale(new_h, new_v)

cv2.imshow('Horizontal', new_h)
cv2.imshow('Vertical', new_v)
cv2.imshow('Multiple', new_mul)
cv2.waitKey(0)
cv2.destroyAllWindows()
