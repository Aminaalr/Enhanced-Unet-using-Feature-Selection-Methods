import tensorflow as tf
import os
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps
import random
import numpy as np 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

#Image Directory Location
pathImg='samplingRandom'
pathMask='maskData'
pathPred='maskPredict'
pathSegment='unetSegment'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
test_ratio=0.2
no_epochs=50
no_image=5

#Loading the Dataset
def loadDataset():
    # the data, shuffled and split between train and test sets   
    imgArr = []
    maskArr = [] 
    nameArr=[]
    dirList=[f for f in listdir(pathImg) if not isfile(join(pathImg, f))]
    print(dirList)    
    for i in range(len(dirList)):
        predClass=os.path.join(pathPred, dirList[i])
        if not os.path.exists(predClass):
            # Create a new directory because it does not exist
            os.makedirs(predClass)
            
        segmentClass=os.path.join(pathSegment, dirList[i])
        if not os.path.exists(segmentClass):
            # Create a new directory because it does not exist
            os.makedirs(segmentClass)
    
        fileList= list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg+'/'+dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i],len(fileList))
        for filename in fileList:
            if (filename.endswith('.jpg')):
                    #try:
                    imgLoad=Image.open(filename)
                    maskLoad=Image.open(str(filename).replace(pathImg,pathMask))
                    #resImg=imgLoad.resize((image_size, image_size),Image.Resampling.BICUBIC)
                    numImg=(np.array(imgLoad))
                    numMask=(np.array(maskLoad))
                    imgArr.append(numImg)
                    maskArr.append(numMask) 
                    nameArr.append(str(filename))
                    #except:
                    #print('Problem in File : ',filename)                
    print(len(imgArr))
    imgArr = np.array(imgArr)
    maskArr = np.array(maskArr)    
    return imgArr, maskArr, nameArr
    
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
    print('\n\nClassName\tTP\t\tFP\tFN\tTN\tPrecision\tSensitivity\tSpecificity\tF-Score\t\tAccuracy')
    for i in range(len(class_names)):
        print(class_names[i]+"\t\t{0:.0f}".format(TP[i])+"\t{0:.0f}".format(FP[i])+"\t{0:.0f}".format(FN[i])+"\t{0:.0f}".format(TN[i])+"\t{0:.4f}".format(PPV[i])+"\t\t{0:.4f}".format(TPR[i])+"\t\t{0:.4f}".format(TNR[i])+"\t\t{0:.4f}".format(FScore[i])+"\t\t{0:.4f}".format(ACC[i]))

def unetModel():
    #Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    r1 = tf.keras.layers.ReLU()(b1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(r1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    r2 = tf.keras.layers.ReLU()(b2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(r2)
 
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    b3 = tf.keras.layers.BatchNormalization()(c3)
    r3 = tf.keras.layers.ReLU()(b3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(r3)
 
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    b4 = tf.keras.layers.BatchNormalization()(c4)
    r4 = tf.keras.layers.ReLU()(b4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(r4)
 
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    b5 = tf.keras.layers.BatchNormalization()(c5)
    r5 = tf.keras.layers.ReLU()(b5)
    c5 = tf.keras.layers.Dropout(0.3)(r5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.ReLU()(u6)
 
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.ReLU()(u7)
 
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.ReLU()(u8)
 
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.ReLU()(u9)
 
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u9) 
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],name="U-Net" )
    return model
    
imgArr, maskArr, nameArr = loadDataset()
x_train, x_test, y_train, y_test =train_test_split(imgArr, maskArr, test_size=test_ratio, random_state=SEED)

f,ax = plt.subplots(no_image,2,figsize=(8,15))
numList=[i for i in range(len(imgArr))]
l = random.sample(numList,no_image)
for i,id_ in enumerate(l):
    ax[i][0].imshow(imgArr[id_])
    msk=cv2.bitwise_not(maskArr[id_]*255)
    ax[i][1].imshow(msk,cmap='binary')
ax[0][0].set_title('Images')
ax[0][1].set_title('Masks')
plt.savefig("unet_input.png")
plt.show() 
 
model=unetModel()
model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()    

model_checkpoint = ModelCheckpoint("weights/best.hdf5", monitor='loss',verbose=1)
model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=100, epochs=no_epochs, callbacks=[model_checkpoint])
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
plt.figure()
plt.plot( loss, label='Training loss')
plt.plot( val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
plt.figure()
plt.plot( accuracy, label='Training accuracy')
plt.plot( val_accuracy, label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

#Load the Saved Model 
model.load_weights("weights/best.hdf5")

#plot the Results
f,ax = plt.subplots(no_image,3,figsize=(16,30))
l = random.sample(numList,no_image)
for i,id_ in enumerate(l):
    msk=cv2.bitwise_not(maskArr[id_]*255)    
    ax[i][0].imshow(msk, cmap='binary')
    img=imgArr[id_]
    prediction = model.predict(img[tf.newaxis, ...])[0]
    predicted_mask = (prediction > 0.5).astype(np.uint8)  
    segment= cv2.bitwise_and(img, img, mask=predicted_mask)
    predicted=cv2.bitwise_not(predicted_mask) 
    ax[i][1].imshow(predicted, cmap='binary') 
    ax[i][2].imshow(segment)    
    
ax[0][0].set_title('Mask')
ax[0][1].set_title('Predicted Mask')
ax[0][2].set_title('Unet-Segment')
plt.savefig("unet_output.png")
plt.show()

#predict the test Dataset
y_pred= model.predict([x_test], batch_size=100)

#Confusion matrix
cm = confusion_matrix(y_test.ravel(), y_pred.ravel()>0.5) 
#Overall Performance 
performance_metrics(cm[:2,:2],['Maks','No-Mask'])   
score, acc = model.evaluate(x_test, y_test, batch_size=100,verbose=1)
print("Test %s: %.2f%%" % (model.metrics_names[1], acc*100))

#Predict & Save the U-net Segmentation Images
for i in range(len(imgArr)):
    img=imgArr[i].astype(np.uint8)
    prediction = model.predict(img[tf.newaxis, ...])[0]
    predicted_mask = (prediction > 0.5).astype(np.uint8)      
    segment= cv2.bitwise_and(img, img, mask=predicted_mask) 
    cv2.imwrite(nameArr[i].replace(pathImg,pathSegment), cv2.cvtColor(segment.astype(np.uint8), cv2.COLOR_RGB2BGR)) 
    predictedmask=cv2.normalize(predicted_mask, None, 0, 1,cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    cv2.imwrite(nameArr[i].replace(pathImg,pathPred), predictedmask) 
    