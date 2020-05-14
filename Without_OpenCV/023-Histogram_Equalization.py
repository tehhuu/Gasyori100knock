import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate #list(accumulate(A))

def Histogram_Equalization(img):
    height, width, color = img.shape
    #ヒストグラムを計算
    hist, _ = np.histogram(img, bins=256)
    #累積話を計算
    hist_acc = np.array(list(accumulate(hist)))

    Z_max = img.max()
    S = height * width * color

    new = np.zeros_like(img, dtype=np.float)
    for h in range(height):
        for w in range(width):
            for c in range(color):
                new[h][w][c] = Z_max / S * hist_acc[img[h][w][c]]
    new = new.astype(np.uint8)

    return new


img = cv2.imread('../assets/imori.jpg')

new = Histogram_Equalization(img)

hist_new = plt.hist(new.ravel(), bins=255, rwidth=0.8, range=(0, 255), label='new')
plt.show(hist_new)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
