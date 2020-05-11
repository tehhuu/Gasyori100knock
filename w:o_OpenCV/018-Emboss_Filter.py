import cv2
import numpy as np

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new


def Emboss_filter(gray):
    H, W = gray.shape
    gray = gray.astype(np.float)
    pad = 1
    pad_gray = np.zeros((H+2*pad, W+2*pad), dtype=np.float)
    pad_gray[pad:H+pad, pad:W+pad] = gray.copy()
    new = pad_gray.copy()

    filt = [[-1, -2, 0], [-1, 1, 1], [0, 1, 2]]
    for i in range(pad, H+pad):
        for j in range(pad, W+pad):
            new[i][j] = np.sum(pad_gray[i-pad:i+pad+1, j-pad:j+pad+1] * filt)
    
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new[pad:H+pad, pad:W+pad]


img = cv2.imread('assets/imori.jpg')

gray = Grayscale(img)
new = Emboss_filter(gray)

cv2.imshow('Emboss', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
