import cv2
import numpy as np
import matplotlib.pyplot as plt

def Grayscale_transformation(img):
    hist_tuple = plt.hist(img.ravel(), bins=255, rwidth=1.0, range=(0, 255), label='original')
    hist_list = np.array(hist_tuple)

    ind = np.where(hist_list[0] > 0)
    min_org = np.min(hist_list[1][ind])
    max_org = np.max(hist_list[1][ind])
    max_new = 255
    min_new = 0
    
    new = (max_new-min_new) / (max_org-min_org) * (img - min_org) + min_new
    new = new.astype(np.uint8)

    return new


img = cv2.imread('../assets/imori_256x256_dark.png')

# 濃度階調変換
new = Grayscale_transformation(img)

# ヒストグラムを表示
plt.hist(new.ravel(), bins=255, rwidth=1.0, range=(0, 255), label='new')
plt.show()

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
