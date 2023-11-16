import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda, Multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import keras_efficientnet_v2
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import random
warnings.filterwarnings("ignore")

SEED=0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'    
    tf.config.threading.set_inter_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()

#Image Directory Location
pathImg='unetSegment'

#Image Size
image_size = 32

#Split Ratio
test_ratio=0.2
dropout_rate = 0.2
no_epochs=50

#Normalization of Data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Loading the Dataset
def loadDataset():
    # the data, shuffled and split between train and test sets   
    imgArr = []
    image_label = []
    class_names = []    
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
                    resImg=imgLoad.resize((image_size, image_size),Image.Resampling.BICUBIC)
                    numImg=(np.array(resImg)).astype('float64')
                    normImg=NormalizeData(numImg)*((i+1)/len(dirList))
                    imgArr.append(normImg)
                    image_label.append(i) 
                    class_names.append(dirList[i])
                except:
                    print('Problem in File : ',filename)                
    print(len(imgArr))
    imgArr = np.array(imgArr)
    classNames = sorted(set(class_names), key=class_names.index)
    labelArr = to_categorical(np.array(image_label))

    #Fix stratified sampling split
    x_train, x_test, y_train, y_test=train_test_split(imgArr, labelArr, test_size=test_ratio, random_state=SEED, stratify=labelArr)
    return x_train, x_test, y_train, y_test, classNames
    
#Performance Matrics    
def performance_metrics(cnf_matrix,class_names):
    #Confusion Matrix Plot
    cmd = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
    cmd.plot(cmap = 'Greens')
    cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
    #Find All Parameters
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # F1-Score accuracy for each class
    FScore = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+TN+FN)
    print('\n\nClassName\tTP\tFP\tFN\tTN\tPrecision\tSensitivity\tSpecificity\tF-Score\t\tAccuracy')
    for i in range(len(class_names)):
        print(class_names[i]+"\t\t{0:.0f}".format(TP[i])+"\t{0:.0f}".format(FP[i])+"\t{0:.0f}".format(FN[i])+"\t{0:.0f}".format(TN[i])+"\t{0:.4f}".format(PPV[i])+"\t\t{0:.4f}".format(TPR[i])+"\t\t{0:.4f}".format(TNR[i])+"\t\t{0:.4f}".format(FScore[i])+"\t\t{0:.4f}".format(ACC[i]))

# load data
x_train, x_test, y_train, y_test, classNames = loadDataset() 
 
#Build Model for efficientNet on U-net Segmentation 
def unet_efficientNet_model(num_class):
    model = keras_efficientnet_v2.EfficientNetV2S(input_shape=(image_size, image_size, 3), num_classes=num_class, classifier_activation='softmax', dropout=0.1)
    print(model.summary())
    return model    

#Model Initial
efficientNet_Model = unet_efficientNet_model(len(classNames))
    
#Model Compile
efficientNet_Model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#check Points
model_checkpoint = ModelCheckpoint(filepath='weights/efficient.hdf5', monitor='loss',verbose=1)

#Start Time
stTime = datetime.now()

#EfficientNet Data Training
history = efficientNet_Model.fit(x_train,y_train,validation_data=(x_test, y_test),epochs=no_epochs, batch_size=100)

#End Time
endTime = datetime.now()

#Total Training Time
trainTime=endTime-stTime
print('Total Training Time : ',trainTime)

#predict the test Dataset
y_pred= efficientNet_Model.predict([x_test], batch_size=100)
    
# list all data in history
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("efficient_accuracy.png")
plt.show()

# summarize history for Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("efficient_loss.png")
plt.show()
    
#Confusion matrix
cm=confusion_matrix(np.argmax(y_test, 1),np.argmax(y_pred, 1))   
#Overall Performance 
performance_metrics(cm,classNames)   
score, acc = efficientNet_Model.evaluate(x_test, y_test, batch_size=100,verbose=1)
print("Test %s: %.2f%%" % (efficientNet_Model.metrics_names[1], acc*100))

