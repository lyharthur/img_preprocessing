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



imglistO = glob( "1. STI_ET Layer(preprocessing)/**/OK/*[0-9].tif" )

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


for i in range(len(imglistO)):

    imgO = cv2.imread(imglistO[i])
    filename = imglistO[i]
    filename = filename[:-4]+'_NEW.tif'
    filename = filename.replace('preprocessing','Hough')
    #print(filename)
    if os.path.isfile(filename):
        #print("delet")
        os.remove(filename)


for i in range(len(imglistO)):

    imgO = cv2.imread(imglistO[i])
    filename = imglistO[i]
    
    filename = filename[:-4]+'_NEW.tif'
    filename = filename.replace('preprocessing','Hough')
    print(filename)

    gray = np.copy(imgO)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((2,2),np.uint8)

    #imgO = cv2.Laplacian(imgO,cv2.CV_8UC1)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    blur = cv2.erode(blur,kernel,1)
    blur = cv2.dilate(blur,kernel2,1)
    
    
    #edges = cv2.Canny(blur,80,160)
    edgestmp = cv2.Canny(blur,90,180)
    edges = np.copy(edgestmp)

    #edges = auto_canny(blur)
    #edgestmp = auto_canny(blur)


    edges = cv2.dilate(edges,kernel,1)
    edgestmp = cv2.dilate(edgestmp,kernel,1)

    #edges = cv2.erode(edges,kernel,1)
    #edgestmp = cv2.erode(edgestmp,kernel,1)



    #imgO = cv2.fastNlMeansDenoising(imgO,None,10,7,21)
    #ret,imgO = cv2.threshold(imgO,150,255,cv2.THRESH_BINARY)
    #imgO = cv2.adaptiveThreshold(imgO,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    lines = cv2.HoughLinesP(edgestmp,1,np.pi/180,60,minLineLength=20,maxLineGap=5)
    
    if lines != None:
        lines1 = lines[:,0,:]
        for x1,y1,x2,y2 in lines1[:]:
            #cv2.line(imgO,(x1,y1),(x2,y2),(0,50,0),3)
            if (x2 - x1 <= 6 and x2 - x1 >= 0)  or (y1 - y2 <= 6 and y1 - y2 >= 0):            
                cv2.line(edgestmp,(x1,y1),(x2,y2),(0),3)

    edgestmp = cv2.erode(edgestmp,kernel2,1)
    edges = cv2.erode(edges,kernel2,1)



    #imgO = cv2.fastNlMeansDenoising(imgO,None,10,7,21)
    #ret,imgO = cv2.threshold(imgO,150,255,cv2.THRESH_BINARY)
    #imgO = cv2.adaptiveThreshold(imgO,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    new_im = Image.new('L', (1440,480))

    img = Image.fromarray(imgO)
    imgE = Image.fromarray(edges)
    imgEt = Image.fromarray(edgestmp)

    new_im.paste(img, (0,0))
    new_im.paste(imgE, (480,0))
    new_im.paste(imgEt, (960,0))
    
    imgEt.save(filename)
    
    #new_im.save(filename)


    #os.rename(filename, filenameN)

    #cv2.imwrite(filenameN,cv2.imread(imglistD[i]))
    #cv2.imwrite(filename,edges)
    print('-------------------------')

print('done')
