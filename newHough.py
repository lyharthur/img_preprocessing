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
import pywt
from skimage.io import imread, imsave

imglistO = glob( "testNEW/**/**/**Class.tif" )
imglistD = glob( "testNEW/**/**/**Defect.tif" )

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def process(image):

    
    gray = image

    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((2,2),np.uint8)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,90,180)
    edges = cv2.dilate(edges,kernel,1)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength=15,maxLineGap=3)
    if lines != None:
        lines1 = lines[:,0,:]
        for x1,y1,x2,y2 in lines1[:]:
            if (x2 - x1 <= 3 and x2 - x1 >= 0)  or (y1 - y2 <= 3 and y1 - y2 >= 0):            
                cv2.line(edges,(x1,y1),(x2,y2),(0),3)
                
    #edgestmp = cv2.dilate(edgestmp,kernel,1)
    #edgestmp = cv2.erode(edgestmp,kernel,1)
    #output = cv2.erode(output,kernel2,1)
    
    return edges


def compare_image(img1 ,img2 ,img3 ):
    output = np.zeros(img1.shape)
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    mean3 = np.mean(img3)
    print(mean1,mean2,mean3)
    if (mean1 - mean3 > 5 or mean2 - mean3 > 5):
        output  = img3
    elif mean3 < 0.5 :
        output  = img3
        for i in range(160,319):
            for j in range(160,319):
                if int(img1[i,j]) + int(img2[i,j])>=255:
                    output[i,j] = 255
                else:
                    output[i,j] = 0
    else :
        output  = img3
        for i in range(160,319):
            for j in range(160,319):
                if int(img1[i,j]) + int(img2[i,j]) + int(img3[i,j]) >=510:
                    output[i,j] = 255
                else:
                    output[i,j] = 0
    return output

def nine_grid(image):
    '''123
       456
       789 '''
    image1 = image[0:159,0:159].sum()    
    image2 = image[0:159,160:319].sum()
    image3 = image[0:159,320:479].sum()
    
    image4 = image[160:319,0:159].sum()
    image5 = image[160:319,160:319].sum()
    image6 = image[160:319,320:479].sum()
    
    image7 = image[320:479,0:159].sum()
    image8 = image[320:479,160:319].sum()
    image9 = image[320:479,320:479].sum()
    count = 0
    if(image5 > image1):
        count += 1
    if(image5 > image2):
        count += 1
    if(image5 > image3):
        count += 1
    if(image5 > image4):
        count += 1
    if(image5 > image6):
        count += 1
    if(image5 > image7):
        count += 1
    if(image5 > image8):
        count += 1
    if(image5 > image9):
        count += 1
    print(count)
    if(count >= 5):
        return False #found
    else:
        return True # not found
        

#code begin
for i in range(len(imglistO)):

    imgO = cv2.imread(imglistO[i])
    
    filename = imglistO[i]
    filename = filename.replace('testNEW','found')

    #print(filename)
    if os.path.isfile(filename):
        #print("delet")
        os.remove(filename)
    filename = filename.replace('found','notfound')
    if os.path.isfile(filename):
        #print("delet")
        os.remove(filename)

for i in range(len(imglistO)):
    filename = imglistO[i]
    imgDD = cv2.imread(imglistD[i])
    imgOO = cv2.imread(imglistO[i])
    imgDD = cv2.cvtColor(imgDD, cv2.COLOR_BGR2GRAY)
    imgOO = cv2.cvtColor(imgOO, cv2.COLOR_BGR2GRAY)   
    #imgO = cv2.imread(filename)
    
    imgO = imread(filename)
    im_array = np.array_split(imgO, 3, axis=2) 

    imgO1 =  im_array[0]
    imgO1 = imgO1[:,:,0]
    final1 = process(imgO1)

    imgO2 =  im_array[1]
    imgO2 = imgO2[:,:,0]
    final2 = process(imgO2)

    imgO3 =  im_array[2]
    imgO3 = imgO3[:,:,0]
    final3 = process(imgO3)
    
    #filename = filename[:-4]+'_NEW.tif'
   
    #print(filename)
    
    if final3.shape != (480,480):
        im_array = np.array_split(imgO, 2, axis=0)
        
        imgO1 =  im_array[0]
        imgO1 = imgO1[0,:,:]
        final1 = process(imgO1)

        imgO2 =  im_array[1]
        imgO2 = imgO2[0,:,:]
        final2 = process(imgO2)
        print(imgO.shape,imgO2.shape,imgO1.shape)
        output = compare_image(final1,final2,final2)
        #output = final1
    else:
        output = compare_image(final1,final2,final3)
        #output = final1


    
    new_im = Image.new("RGB", (1440,1440),(100,120,200))

    imgO = Image.fromarray(imgO1)
    img  = Image.fromarray(output)
    img1 = Image.fromarray(final1)    
    img2 = Image.fromarray(final2)
    img3 = Image.fromarray(final3)

    

    new_im.paste(imgO, (480,0))

    new_im.paste(img, (480,960)) #trus final
    
    new_im.paste(img1, (0,480))    
    new_im.paste(img2, (480,480))
    new_im.paste(img3, (960,480))
    

    if (np.array_equal(imgO1 ,imgDD) or (np.mean(output[180:300,180:300])<2) and nine_grid(output)):
        filename = filename.replace('testNEW','notfound')
        new_im.save(filename)
        #cv2.imwrite(filename,imgOO)
    else:
        filename = filename.replace('testNEW','found')
        new_im.save(filename)
        #cv2.imwrite(filename,imgOO)
        
        


    
    #output.save(filename)
    #new_im.save(filename)


    #os.rename(filename, filenameN)
    print(filename)
    #cv2.imwrite(filenameN,cv2.imread(imglistD[i]))
    #cv2.imwrite(filename,output)
    print('-------------------------')
    


print('done')
