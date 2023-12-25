import os
from os import listdir
from os.path import isfile, join
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import cv2
import warnings
warnings.filterwarnings("ignore")

#Define the Cluster number
k = 2

#Image Directory Location
maskPath='maskData'

#Sampling Directory
dirPath='samplingRandom'

def maskImages():
    #Loading the Dataset
    dirList=[f for f in listdir(dirPath) if not isfile(join(dirPath, f))]
    print(dirList)    
    for i in range(len(dirList)):
        maskClass=os.path.join(maskPath, dirList[i])
        if not os.path.exists(maskClass):
            # Create a new directory because it does not exist
            os.makedirs(maskClass)
            
        fileList= list()
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(dirPath, dirList[i])):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i],len(fileList))
        for filename in fileList:
            if (filename.endswith('.jpg')):
                try:
                    #Read Image
                    image = cv2.imread(filename)
                    
                    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
                    pixel_vals = image.reshape((-1,3))
 
                    # Convert to float type
                    pixel_vals = np.float32(pixel_vals)

                    #the below line of code defines the criteria for the algorithm to stop running,
                    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
                    #becomes 100%
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1)
 
                    # then perform k-means clustering with number of clusters defined as K
                    #also random centres are initially choosed for k-means clustering
                    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
                    # convert data into 8-bit values
                    centers = np.uint8(centers)
                    mask_data = centers[labels.flatten()]
 
                    # reshape data into the original image dimensions
                    mask_image = mask_data.reshape((image.shape))
                    
                    #Grey Image
                    grey_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                    
                    #Invert Mask
                    inv_mask=cv2.bitwise_not(grey_image)

                    # normalize the binary image
                    mask_binary_image = cv2.normalize(inv_mask, None, 0, 1,cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
                    
                    #save ImageFile
                    cv2.imwrite(filename.replace(dirPath, maskPath),  mask_binary_image)                    
                except:
                    print('Problem in File : ',filename)        
    
maskImages()
  
