import cv2
import numpy as np
import matplotlib.pyplot as plt

def Change_std_mean(img, mean_new, std_new):
    m = img.mean()
    s = img.std()

    new = std_new * (img - m) / s + mean_new
    np.clip(img, 0, 255)
    new = new.astype(np.uint8)

    return new

img = cv2.imread('assets/imori_256x256_dark.png')

#平均と標準偏差を変更
new = Change_std_mean(img, 128, 52)

# ヒストグラムを表示
plt.hist(new.ravel(), bins=255, rwidth=0.8, range=(0, 255), label='new')
plt.show()

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
