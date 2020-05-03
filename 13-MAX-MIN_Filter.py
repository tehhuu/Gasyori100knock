import cv2
import numpy as np

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new

def MAX_MIN(img, size):
    H, W = img.shape
    pad = size //2

    pad_img = np.zeros((H + 2*pad, W + 2*pad), dtype=np.float)
    pad_img[pad:pad+H, pad:pad+W] = img.copy()
    new = pad_img.copy()

    for i in range(pad, pad+H):
        for j in range(pad, pad+W):
            filt = pad_img[i-pad:i+pad+1, j-pad:j+pad+1]
            new[i, j] = np.max(filt) - np.min(filt)

    new = new.astype(np.uint8)
    return new[pad:pad+H, pad:pad+W]


img = cv2.imread('assets/imori.jpg')

gray = Grayscale(img)
new = MAX_MIN(gray, 3)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
