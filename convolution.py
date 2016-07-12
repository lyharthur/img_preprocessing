import numpy as np
from scipy import signal
import cv2
from PIL import Image
from glob import glob
imglistR = glob( "1. STI_ET Layer/**/OK/**REF.tif" )
imglistD = glob( "1. STI_ET Layer/**/OK/**DEF.tif" )

def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   return signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

def split_quater(img1,img2,n):
   iq1 = img1
   iq2 = img2
   if n == 1:
      iq1 = np.copy(iq1[0:239,0:239])
      iq2 = np.copy(iq2[0:239,0:239])
   elif n == 2:
      iq1 = np.copy(iq1[0:239,240:479])
      iq2 = np.copy(iq2[0:239,240:479])
   elif n == 3:
      iq1 = np.copy(iq1[240:479,0:239])
      iq2 = np.copy(iq2[240:479,0:239])
   elif n == 4:
      iq1 = np.copy(iq1[240:479,240:479])
      iq2 = np.copy(iq2[240:479,240:479])
   elif n == 5:
      iq1 = img1
      iq2 = img2
   iq1 = cv2.GaussianBlur(iq1,(5,5),0)
   iq2 = cv2.GaussianBlur(iq2,(5,5),0)
   
   corr_img11 = cross_image(iq1,iq1)
   corr_img12 = cross_image(iq1,iq2)
   y_self,x_self = np.unravel_index(np.argmax(corr_img11), corr_img11.shape)
   y,x = np.unravel_index(np.argmax(corr_img12), corr_img12.shape)

   y_diff = y - y_self
   x_diff = x - x_self
   print(y_diff,x_diff)
   return (y_diff,x_diff)



for i in range(len(imglistD)):
   img1 = cv2.imread(imglistR[i])
   img2 = cv2.imread(imglistD[i])

   #y1,x1 =  split_quater(img1,img2,1)
   #y2,x2 =  split_quater(img1,img2,2)
   #y3,x3 =  split_quater(img1,img2,3)
   #y4,x4 =  split_quater(img1,img2,4)
   y5,x5 =  split_quater(img1,img2,5)
   #y_avg = (y1+y2+y3+y4+y5)//5
   #x_avg = (x1+x2+x3+x4+x5)//5
   #print(y_avg,x_avg)
   y_avg = y5
   x_avg = x5

   
   #shift
   M = np.float32([[1,0,-x_avg],[0,1,-y_avg]])
   img1new = cv2.warpAffine(img1,M,(480,480))

   img1new = cv2.cvtColor(img1new, cv2.COLOR_BGR2GRAY)
   img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   img1new = cv2.GaussianBlur(img1new,(5,5),0)
   img2 = cv2.GaussianBlur(img2,(5,5),0)
   #img1new = cv2.Canny(img1new,100,200)
   #img2 = cv2.Canny(img2,100,200)

   ret,img1new = cv2.threshold(img1new,150,255,cv2.THRESH_BINARY)
   ret,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY)

   #img1new = cv2.adaptiveThreshold(img1new,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
   #img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

   img1new = img2 - img1new
   img1new = cv2.fastNlMeansDenoising(img1new,None,10,7,21)
   
   filename = imglistR[i]
   filename = filename.replace('REF','NEW')
   print(filename)
   cv2.imwrite(filename,img1new)
