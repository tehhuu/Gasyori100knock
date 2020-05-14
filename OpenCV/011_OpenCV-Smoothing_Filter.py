import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

def Smoothing_filter(img, size):
    pixel_size = len(img)
    ratio = pixel_size // size
    
    kernel = np.ones((size, size))
    kernel /= (size)**2

    new = cv2.filter2D(img, -1, kernel)

    return new
            

img = cv2.imread('../assets/imori_noise.jpg')

new = Smoothing_filter(img, 5)
#new = cv2.blur(img, (5, 5)) 

cv2.imshow('', new)
cv2.waitKey(0)
cv2.destroyAllWindows()