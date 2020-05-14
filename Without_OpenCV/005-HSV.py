import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

def RGBtoHSV(img):
    img = img.astype(np.float32)
    img = img.copy() / 255
    G = img[:, :, 0].copy()
    B = img[:, :, 1].copy()
    R = img[:, :, 2].copy()

    Max = np.max(img, axis=2).copy()
    Min = np.min(img, axis=2).copy()
    arg_min = np.argmin(img, axis=2)
    HSV = np.zeros_like(img, dtype=np.float32)

    HSV[..., 0][np.where(Max == Min)] = 0
    ind_1 = np.where(arg_min == 0)
    HSV[..., 0][ind_1] = (60 * (G[ind_1]-R[ind_1])) / (Max[ind_1]-Min[ind_1]) + 60
    ind_2 = np.where(arg_min == 1)
    HSV[..., 0][ind_2] = (60 * (R[ind_2]-G[ind_2])) / (Max[ind_2]-Min[ind_2]) + 300
    ind_3 = np.where(arg_min == 2)
    HSV[..., 0][ind_3] = (60 * (G[ind_3]-B[ind_3])) / (Max[ind_3]-Min[ind_3]) + 180
    HSV[..., 2] = Max.copy()
    HSV[..., 1] = Max.copy() - Min.copy()

    return HSV


def HSVtoRGB(HSV):
    C = HSV[..., 1].copy()
    H_ = HSV[..., 0].copy() / 60
    X = C * (1 - np.abs(H_%2 - 1))
    V_C = HSV[:, :, 2] - C

    new = np.dstack([V_C, V_C])
    new = np.dstack([new, V_C])

    cx_pos = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]]
    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        new[..., cx_pos[i][0]][ind] += C[ind]
        new[..., cx_pos[i][1]][ind] += X[ind]

    new *= 255
    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    new = new[..., ::-1]
    return new


img = cv2.imread('../assets/imori.jpg')

hsv = RGBtoHSV(img) #HSV変換
hsv[..., 0] = (hsv[..., 0] + 180) % 360 #色相Hを反転
new = HSVtoRGB(hsv) #RGBに戻す

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()