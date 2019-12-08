import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from preprocessing import binary_otsus, deskew
from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import os.path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings



#%matplotlib inline
def whiteBlackRatio(img):
    h = img.shape[0]
    w = img.shape[1]
    #initialized at 1 to avoid division by zero
    blackCount=1
    whiteCount=0
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1
            else:
                whiteCount+=1
    return whiteCount/blackCount

def blackPixelsCount(img):
    blackCount=1 #initialized at 1 to avoid division by zero when we calculate the ratios
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1
            
    return blackCount
def horizontalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for y in range(0,h):
        prev=img[y,0]
        transitions=0
        for x in range (1,w):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)
            
    return maximum
def verticalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for x in range(0,w):
        prev=img[0,x]
        transitions=0
        for y in range (1,h):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)
            
    return maximum

def histogramAndCenterOfMass(img):
    h = img.shape[0]
    w = img.shape[1]
    histogram=[]
    sumX=0
    sumY=0
    num=0
    for x in range(0,w):
        localHist=0
        for y in range (0,h):
            if(img[y,x]==0):
                sumX+=x
                sumY+=y
                num+=1
                localHist+=1
        histogram.append(localHist)
      

    return sumX/num , sumY/num, histogram

def getFeatures(img):
    x,y= img.shape
    featuresList=[]
    # first feature: height/width ratio
    featuresList.append(y/x)
    #second feature is ratio between black and white count pixels
    featuresList.append(whiteBlackRatio(img))
    #third and fourth features are the number of vertical and horizontal transitions
    featuresList.append(horizontalTransitions(img))
    featuresList.append(verticalTransitions(img))

    #print (featuresList)
    #splitting the image into 4 images
    topLeft=img[0:y//2,0:x//2]
    topRight=img[0:y//2,x//2:x]
    bottomeLeft=img[y//2:y,0:x//2]
    bottomRight=img[y//2:y,x//2:x]

    #get white to black ratio in each quarter
    featuresList.append(whiteBlackRatio(topLeft))
    featuresList.append(whiteBlackRatio(topRight))
    featuresList.append(whiteBlackRatio(bottomeLeft))
    featuresList.append(whiteBlackRatio(bottomRight))

    #the next 6 features are:
    #• Black Pixels in Region 1/ Black Pixels in Region 2.
    #• Black Pixels in Region 3/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 3.
    #• Black Pixels in Region 2/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 4
    #• Black Pixels in Region 2/ Black Pixels in Region 3.
    topLeftCount=blackPixelsCount(topLeft)
    topRightCount=blackPixelsCount(topRight)
    bottomLeftCount=blackPixelsCount(bottomeLeft)
    bottomRightCount=blackPixelsCount(bottomRight)

    featuresList.append(topLeftCount/topRightCount)
    featuresList.append(bottomLeftCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomLeftCount)
    featuresList.append(topRightCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomRightCount)
    featuresList.append(topRightCount/bottomLeftCount)
    #get center of mass and horizontal histogram
    xCenter, yCenter,xHistogram =histogramAndCenterOfMass(img)
    featuresList.append(xCenter)
    featuresList.append(yCenter)
    featuresList.extend(xHistogram)
    #print(len(featuresList))
    return featuresList
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        
def main():
    data=np.array([])
    classes=np.array([])
    chars=get_immediate_subdirectories('../binary')
    count=0
    numOfFeatures=216
    for char in chars:
        listOfFiles = getListOfFiles('../binary/'+char)
        
        for filename in listOfFiles:
            img = cv.imread(filename)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            binary_img = binary_otsus(gray_img, 0)
            features=getFeatures(binary_img)
            data= np.append(data,features)
            classes=np.append(classes,char)
            count+=1
            #print (features)
            #show_image(img)
    #np.savetxt('test.txt', data)    
    

    print(data.shape)
    print(classes.shape)
    data=np.reshape(data,(count,numOfFeatures))
    print(data.shape)
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size = 0.10)
    svclassifier = SVC(kernel='rbf', gamma =0.01 , C =100)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    test1=cv.imread(filename)
    print(classification_report(y_test, y_pred))

main()

