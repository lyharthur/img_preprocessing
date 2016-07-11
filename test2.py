from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
rowN = 300
colN = 240
img1 = cv2.imread("d8_P4PB25#03_DEF.tif")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = img1[0:240,0:240]
img1 = cv2.GaussianBlur(img1,(5,5),0)
cv2.imwrite('12.jpg',img1)

img2 = cv2.imread("d8_P4PB25#03_REF.tif")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = img2[0:150,0:150]
template = cv2.GaussianBlur(img2,(5,5),0)
cv2.imwrite('22.jpg',img2)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img1,template,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum


top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img1,top_left, bottom_right, 255, 2)

plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img1,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('cv2.TM_CCOEFF_NORMED')
plt.show()







print("done")
