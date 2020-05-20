import cv2
import numpy as np


def Nearest_Neighbor(img, mag):
    H, W, C = img.shape
    H_new, W_new = int(H*mag), int(W*mag)
    new = np.zeros((H_new, W_new, C), dtype=np.uint8)

    for i in range(H_new):
        for j in range(W_new):
            new[i, j, :] = img[round(i/mag), round(j/mag), :]
    
    return new


img = cv2.imread('../assets/imori.jpg')

new = Nearest_Neighbor(img, 1.5)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
