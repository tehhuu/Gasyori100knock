import cv2
import numpy as np

def GaussianFilter(img, matrix):
    H, W, _ = img.shape
    img = img.astype(np.float32)
    size = len(matrix)
    di = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    dj = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    new = img.copy()

    for i in range(H):
        for j in range(W):
            pixel = np.zeros((size, size, 3), dtype=np.float32)
            filt = np.zeros_like(matrix, dtype=np.float32)
            for k in range(3):
                for l in range(3):
                    y, x = i+di[k][l], j+dj[k][l]
                    if 0<= y < H and 0<= x < W:
                        filt[k, l] = matrix[k, l]
                        pixel[k, l, :] = img[y, x, :]
            
            filt = np.multiply(filt, filt.sum())
            for c in range(3):
                new[i, j, c] = np.sum(np.multiply(pixel[..., c], filt))

    new = np.clip(new, 0, 255)
    new = new.astype(np.uint8)
    return new


img = cv2.imread('assets/imori_noise.jpg')
matrix = np.multiply([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], float(1/16))

new = GaussianFilter(img, matrix)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()