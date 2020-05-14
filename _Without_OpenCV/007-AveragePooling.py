import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

def AveragePooling(img, size):
    pixel_size = len(img)
    ratio = pixel_size // size
    
    new = np.zeros_like(img, dtype=np.float32)

    for i in range(ratio):
        for j in range(ratio):
            for c in range(3):
                new[i*size:(i+1)*size, j*size:(j+1)*size, c] = np.mean(img[i*size:(i+1)*size, j*size:(j+1)*size, c])

    new = new.astype(np.uint8)
    return new
            

img = cv2.imread('../assets/imori.jpg')

new = AveragePooling(img, 8)

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()