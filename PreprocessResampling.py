import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageOps
from imblearn.over_sampling import RandomOverSampler
from numpy import asarray
import warnings
warnings.filterwarnings("ignore")

#Image Directory Location
pathImg='images'

#Sampling Directory
dirPath='samplingRandom'

#Image Size
image_size = 32

# Learning for Imbalanced Classification
def OverSample(imgArr, labelArr):
   #Array shape for Reshape
    x1=imgArr.shape[1]
    x2=imgArr.shape[2]
    x3=imgArr.shape[3]
    
    #Oversampling Strategy    
    strategy = {0:5000, 1:20000, 2:5000, 3:12000, 4:28000, 5:6000, 6:5000, 7:5000, 8:5000, 9:5000, 10:27000, 11:5000, 12:5000, 13:8000, 14:10000, 15:30000, 16:5000, 17:5000, 18:5000, 19:8000, 20:12000}
    
    oversample = RandomOverSampler(sampling_strategy=strategy)
    
    #Reshape    
    imgArr = (imgArr.reshape(imgArr.shape[0], x1 * x2 * x3))
    
    #Over Sampling
    imgArr, labelArr = oversample.fit_resample(imgArr, labelArr)
    
    #Reshape
    imgArr = (imgArr.reshape(imgArr.shape[0], x1, x2, x3))
    return imgArr, labelArr


def overSampling():
    imgArr = []
    labelArr = [] 
    class_names = [] 
    #Loading the Dataset
    dirList=[f for f in listdir(pathImg) if not isfile(join(pathImg, f))]
    print(dirList)    
    for i in range(len(dirList)):
        fileList= list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg+'/'+dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i],len(fileList))
        for filename in fileList:
            if (filename.endswith('.jpg')):
                try:
                    imgLoad=Image.open(filename)
                    resImg=imgLoad.resize((image_size, image_size),Image.BICUBIC)
                    imgArr.append(np.array(resImg))
                    labelArr.append(i) 
                    if dirList[i] not in class_names:
                        class_names.append(dirList[i])
                except:
                    print('Problem in File : ',filename)        

    # Over Sample
    imgArr = np.array(imgArr)
    imgArr, labelArr = OverSample(imgArr, labelArr) 
    if not os.path.exists(dirPath):
        # Create a new directory because it does not exist
        os.makedirs(dirPath)
    for className in class_names:
        classDirPath=os.path.join(dirPath, className)
        if not os.path.exists(classDirPath):
            os.makedirs(classDirPath)   
                
    # Save ReSampling Images as JPEG Files            
    fileCount = [0] * len(class_names)
    for i in range(len(imgArr)):        
        fileCount[labelArr[i]] += 1
        fileName=class_names[labelArr[i]]+'_'+str(fileCount[labelArr[i]])+'.jpg'
        fileName=os.path.join(dirPath, class_names[labelArr[i]],fileName)
        Image.fromarray(imgArr[i]).save(fileName)
    
overSampling()
  
