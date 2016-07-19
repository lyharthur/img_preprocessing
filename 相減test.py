import numpy as np
from scipy import signal
import cv2
from PIL import Image
from glob import glob
import re
import os
import math
import operator
from skimage import data,feature
import skimage.transform as st

kernel = np.ones((3,3),np.uint8)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = np.sum(im1.astype('float'))
   im2_gray = np.sum(im2.astype('float'))

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   return signal.fftconvolve(im1, im2[::-1,::-1], mode='same')

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

    #cv2.imwrite('corr_img11.jpg',corr_img11)
    #cv2.imwrite('corr_img12.jpg',corr_img22)


    y_self,x_self = np.unravel_index(np.argmax(corr_img11), corr_img11.shape)
    y,x = np.unravel_index(np.argmax(corr_img12), corr_img12.shape)

    y_diff = y - y_self
    x_diff = x - x_self
    print(y_diff,x_diff)
    return (y_diff,x_diff)



imglistR = glob( "test/**/**/**Ref.tif" )
imglistD = glob( "test/**/**/**Defect.tif" )
imglistO = glob( "test/**/**/**Class.tif" )


for i in range(len(imglistD)):
    img1 = cv2.imread(imglistD[i])
    img2 = cv2.imread(imglistR[i])
    imgO = cv2.imread(imglistO[i])
    kernel = np.ones((3,3),np.uint8)


    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    imgO = cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY)





    #img1new = cv2.Laplacian(img1new,cv2.CV_64F)
    #img2 = cv2.Laplacian(img2,cv2.CV_64F)
    #imgO = cv2.Laplacian(imgO,cv2.CV_64F)


    imgnew = cv2.GaussianBlur(img1,(7,7),0)
    img2 = cv2.GaussianBlur(img2,(7,7),0)

    #imgnew = cv2.Canny(imgnew,120,240)
    #img2 = cv2.Canny(img2,120,240)

    imgnew = auto_canny(imgnew)
    img2 = auto_canny(img2)

    imgnew = cv2.dilate(imgnew,kernel,1)
    img2 = cv2.dilate(img2,kernel,1)
    #imgnew = cv2.erode(imgnew,kernel,1)
    #img2 = cv2.erode(img2,kernel,1)




    #y5,x5 =  split_quater(imgnew,img2,5)

    """if y5 <10 and y5>-10 and x5>-10 and x5<10:
      y_avg = y5
      x_avg = x5

      #shift
      M = np.float32([[1,0,-x_avg],[0,1,-y_avg]])
      imgnew = cv2.warpAffine(imgnew,M,(480,480))"""



    #ret,img1 = cv2.threshold(img1,150,255,cv2.THRESH_BINARY)
    #ret,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY)
    #ret,imgO = cv2.threshold(imgO,150,255,cv2.THRESH_BINARY)

    #img1new = cv2.adaptiveThreshold(img1new,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



    #detect shift
    #y1,x1 =  split_quater(img1,img2,1)
    #y2,x2 =  split_quater(img1,img2,2)
    #y3,x3 =  split_quater(img1,img2,3)
    #y4,x4 =  split_quater(img1,img2,4)
    #y5,x5 =  split_quater(img1,img2,5) 
    #y_avg = (y1+y2+y3+y4+y5)//5
    #x_avg = (x1+x2+x3+x4+x5)//5
    #print(y_avg,x_avg)

    imgnew = imgnew - img2

    lines = cv2.HoughLinesP(imgnew,1,np.pi/180,100,minLineLength=20,maxLineGap=5)
    if lines != None:
        lines1 = lines[:,0,:]#提取为二维
        for x1,y1,x2,y2 in lines1[:]: 
            cv2.line(imgnew,(x1,y1),(x2,y2),(0,0,0),2)   

    imgnew = cv2.erode(imgnew,kernel,1)
   
   #img1new = cv2.fastNlMeansDenoising(img1new,None,10,7,21)

    print("mean",imgnew.mean())

    if np.array_equal(imgO ,img1):
        same = True
        print('t')
    elif imgnew.mean()< 1 and  hash_value < 2  :
        same = True
        print('t')
    else:
        same = False
        print('f')
   
    filename = imglistD[i]
    filename = filename.replace('Defect','NEW')
   
##    if same:
##        #filenameN = re.sub(r"test\\Lot [0-9]*\\",'test/notfound/',filename)
##        filenameN = filename.replace('test','notfound')
##    else :
##        #filenameN = re.sub(r"test\\Lot [0-9]*\\",'test/found/',filename)
##        filenameN = filename.replace('test','found')
##   
##    print(filenameN)

    #os.rename(filename, filenameN)

    #cv2.imwrite(filenameN,cv2.imread(imglistD[i]))
    cv2.imwrite(filename,imgnew)
    print('-------------------------')

print('done')
