import cv2
import numpy as np

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new


def h_Sobel_filter(gray):
    H, W = gray.shape
    gray = gray.astype(np.float)
    pad = 1
    pad_gray = np.zeros((H+2*pad, W+2*pad), dtype=np.float)
    pad_gray[pad:H+pad, pad:W+pad] = gray.copy()
    new = pad_gray.copy()

    filt = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    for i in range(pad, H+pad):
        for j in range(pad, W+pad):
            new[i][j] = np.sum(pad_gray[i-pad:i+pad+1, j-pad:j+pad+1] * filt)

    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new[pad:H+pad, pad:W+pad]

def v_Sobel_filter(gray):
    H, W = gray.shape
    gray = gray.astype(np.float)
    pad = 1
    pad_gray = np.zeros((H+2*pad, W+2*pad), dtype=np.float)
    pad_gray[pad:H+pad, pad:W+pad] = gray.copy()
    new = pad_gray.copy()

    filt = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    for i in range(pad, H+pad):
        for j in range(pad, W+pad):
            new[i][j] = np.sum(pad_gray[i-pad:i+pad+1, j-pad:j+pad+1] * filt)

    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new[pad:H+pad, pad:W+pad]


def synthesize_grayscale(img1, img2):
    _img1 = img1.copy().astype(np.float)
    _img2 = img2.copy().astype(np.float)
    new = np.sqrt(_img1**2 + _img2**2)
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new


img = cv2.imread('../assets/imori.jpg')

gray = Grayscale(img)
new_h = h_Sobel_filter(gray)
new_v = v_Sobel_filter(gray)
#横方向と縦方向を合成してみた
new_mul = synthesize_grayscale(new_h, new_v)

cv2.imshow('Horizontal', new_h)
cv2.imshow('Vertical', new_v)
cv2.imshow('Multiple', new_mul)
cv2.waitKey(0)
cv2.destroyAllWindows()
