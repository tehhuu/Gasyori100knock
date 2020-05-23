import cv2
import numpy as np
import math

def Bi_linear(img, mag):
    H, W, C = img.shape
    H_new, W_new = int(H*mag), int(W*mag)
    new = np.zeros((H_new, W_new, C), dtype=float)

    for i in range(H_new):
        for j in range(W_new):
            i_, j_ = i / mag, j / mag
            i_org, j_org = math.floor(i_), math.floor(j_)
            di, dj = i_ - i_org, j_ - j_org
            if i_org+1 < H and j_org+1 < W:
                new[i, j] = img[i_org, j_org] * (1-di) * (1-dj) \
                                + img[i_org+1, j_org] * di * (1-dj) \
                                + img[i_org, j_org+1] * (1-di) * dj \
                                + img[i_org+1, j_org+1] * di * dj
            else:
                new[i, j, :] = img[i_org, j_org, :]
    
    new = new.astype(np.uint8)
    return new


img = cv2.imread('../assets/imori.jpg')

new = Bi_linear(img, 1.5)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
