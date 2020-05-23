import cv2
import numpy as np
import math

def Bi_Cubic(img, mag):

    def calc_h_4(l, a=-1):
        new = [0] * 4
        for i in range(4):
            if abs(a * l[i]) <= 1:
                new[i] = (a+2) * np.power(l[i], 3) - (a+3) * np.power(l[i], 2) + 1
            elif 1 < abs(a * l[i]) <= 2:
                new[i] = a * np.power(l[i], 3) - 5 * a * np.power(l[i], 2) + 8 * a * l[i] - 4 * a
            else:
                new[i] = 0
        return new

    H, W, C = img.shape
    H_new, W_new = int(H*mag), int(W*mag)
    new = np.zeros((H_new, W_new, C), dtype=np.float32)

    for i in range(H_new):
        for j in range(W_new):
            i_, j_ = i / mag, j / mag
            i_org, j_org = math.floor(i_), math.floor(j_)
            if 1 < i_org < H-2 and 1 < j_org < W-2:
                di = [abs(i_ - (i_org+x)) for x in range(-1, 3)]
                dj = [abs(j_ - (j_org+y)) for y in range(-1, 3)]
                hi, hj = calc_h_4(di), calc_h_4(dj)
                sum_h = 0
                for y in range(4):
                    for x in range(4):
                        sum_h += hi[x] * hj[y]
                        new[i, j] += img[i_org+x-1, j_org+y-1] * hi[x] * hj[y]
                new[i, j] /= sum_h
            else:
                new[i, j] = img[i_org, j_org]
    
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new


img = cv2.imread('../assets/imori.jpg')

new = Bi_Cubic(img, 1.5)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()
