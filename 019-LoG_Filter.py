import cv2
import numpy as np

def Grayscale(img):
    new = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    new = new.astype(np.uint8)
    return new


def LoG_filter(gray, sigma, size):
    H, W = gray.shape
    gray = gray.astype(np.float)
    pad = size // 2
    pad_gray = np.zeros((H+2*pad, W+2*pad), dtype=np.float)
    pad_gray[pad:H+pad, pad:W+pad] = gray.copy()
    new = pad_gray.copy()

    # フィルタ内の値の計算
    def calc(x, y):
        return (x**2 + y**2 - 2*(sigma**2)) / (2*np.pi*(sigma**6)) * np.exp(((-1)*(x**2+y**2))/(2*(sigma**2)))

    # フィルタの作成
    filt = np.array([[calc(i, j) for j in range((-1)*pad, pad+1)] for i in range((-1)*pad, pad+1)])
    filt /= filt.sum()

    for i in range(pad, H+pad):
        for j in range(pad, W+pad):
            new[i][j] = np.sum(pad_gray[i-pad:i+pad+1, j-pad:j+pad+1] * filt)
    
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new[pad:H+pad, pad:W+pad]


img = cv2.imread('assets/imori_noise.jpg')

gray = Grayscale(img)
new = LoG_filter(gray, 3., 5)

cv2.imshow('LoG', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
