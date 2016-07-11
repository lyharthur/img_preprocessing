from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
img1 = cv2.imread("d8_P4PB25#03_DEF.tif")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.GaussianBlur(img1,(5,5),0)
img1 = cv2.Canny(img1,100,200)


img1a = np.copy(img1[0:239,0:239])
img1b = np.copy(img1[0:239,240:479])
img1c = np.copy(img1[240:479,0:239])
img1d = np.copy(img1[240:479,240:479])

cv2.imwrite('10.jpg',img1b)
img2 = cv2.imread("d8_P4PB25#03_REF.tif")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img2,(5,5),0)
img2 = cv2.Canny(img2,100,200)


img2a = np.copy(img2[0:239,0:239])
img2b = np.copy(img2[0:239,240:479])
img2c = np.copy(img2[240:479,0:239])
img2d = np.copy(img2[240:479,240:479])

cv2.imwrite('20.jpg',img2b)


corr = signal.correlate2d(img1a, img2a, boundary='symm', mode='same')
y_a, x_a = np.unravel_index(np.argmax(corr), corr.shape) # find the match
print(y_a)
print(x_a)

corr = signal.correlate2d(img1b, img2b, boundary='symm', mode='same')
y_b, x_b = np.unravel_index(np.argmax(corr), corr.shape) # find the match
print(y_b)
print(x_b)

corr = signal.correlate2d(img1c, img2c, boundary='symm', mode='same')
y_c, x_c = np.unravel_index(np.argmax(corr), corr.shape) # find the match
print(y_c)
print(x_c)

corr = signal.correlate2d(img1d, img2d, boundary='symm', mode='same')
y_d, x_d = np.unravel_index(np.argmax(corr), corr.shape) # find the match
print(y_d)
print(x_d)

(y_a + y_b + y_c + y_d)//4 - 120

(x_a + x_b + x_c + x_d)//4 - 120


print((y_a + y_b + y_c + y_d)//4 - 120)
print((x_a + x_b + x_c + x_d)//4 - 120)


#print(img1[y-rowN//2 + 1,x-colN//2 + 1]) #左上角的點



print("done")
