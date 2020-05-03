import cv2
import numpy as np

def SmoothingFilter(img, size):
    H, W, C = img.shape
    pad = size //2
    _img = img.astype(np.float)

    #入力画像の画素値は全て整数なので、適当な少数でパデイングをし、平均値の計算では小数を省いて行う。
    pad_img = np.full((H + 2*pad, W + 2*pad, 3), 0.5, dtype=np.float)
    pad_img[pad:pad+H, pad:pad+W, :] = _img.copy()
    new = pad_img.copy()

    for i in range(pad, pad+H):
        for j in range(pad, pad+W):
            filt = pad_img[i-pad:i+pad+1, j-pad:j+pad+1, :].copy()
            for color in range(C):
                new[i, j, color] = np.mean(filt[..., color][filt[..., color] != 0.5])
    new = new.astype(np.uint8)

    return new[pad:pad+H, pad:pad+W, :]


img = cv2.imread('assets/imori.jpg')

new = SmoothingFilter(img, 3)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
