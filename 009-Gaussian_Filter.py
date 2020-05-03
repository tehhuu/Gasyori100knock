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


'''
matrix_2 = np.zeros_like(matrix, dtype=np.float)

def Gause(x, y):
    #return 1 / (2*np.pi*(1.3**2)) * np.exp(((-1)*(x**2+y**2))/(2*(1.3**2)))
    return np.exp(((-1)*(x**2+y**2))/(2*(1.3**2)))

for i in range(-1, 2):
    for j in range(-1, 2):
        matrix_2[i+1, j+1] = Gause(i, j)
matrix_2 /= (2 * np.pi * 1.3 * 1.3)
matrix_2 /= matrix_2.sum()
print(matrix_2)

K_size = 3
sigma = 1.3
pad = 1
K = np.zeros((K_size, K_size), dtype=np.float)
for x in range(-pad, -pad + K_size):
    for y in range(-pad, -pad + K_size):
        K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
K /= (2 * np.pi * sigma * sigma)
K /= K.sum()
print(K)
'''