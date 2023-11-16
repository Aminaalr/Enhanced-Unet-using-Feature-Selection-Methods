from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate,Conv2DTranspose, BatchNormalization, DepthwiseConv2D
from keras.models import Model
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda, Multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import keras_efficientnet_v2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.experimental import preprocessing
from keras.layers import Add
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2,ResNet50
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

warnings.filterwarnings("ignore")

SEED = 0


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)


# Call the above function with seed value
set_global_determinism(seed=SEED)


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

# Image Directory Location
pathImg = 'unetSegment'

# Image Size
image_size = 32

# Split Ratio
test_ratio = 0.2
dropout_rate = 0.2
no_epochs = 50


# Normalization of Data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Loading the Dataset
def loadDataset():
    # the data, shuffled and split between train and test sets
    imgArr = []
    image_label = []
    class_names = []
    dirList = [f for f in listdir(pathImg) if not isfile(join(pathImg, f))]
    print(dirList)
    for i in range(len(dirList)):
        fileList = list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg + '/' + dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i], len(fileList))
        for filename in fileList:
            if (filename.endswith('.jpg')):
                try:
                    imgLoad = Image.open(filename)
                    resImg = imgLoad.resize((image_size, image_size), Image.Resampling.BICUBIC)
                    numImg = (np.array(resImg)).astype('float64')
                    normImg = NormalizeData(numImg) * ((i + 1) / len(dirList))
                    imgArr.append(normImg)
                    image_label.append(i)
                    class_names.append(dirList[i])
                except:
                    print('Problem in File : ', filename)
    print(len(imgArr))
    imgArr = np.array(imgArr)
    classNames = sorted(set(class_names), key=class_names.index)
    labelArr = to_categorical(np.array(image_label))

    # Fix stratified sampling split
    x_train, x_test, y_train, y_test = train_test_split(imgArr, labelArr, test_size=test_ratio, random_state=SEED,
                                                        stratify=labelArr)
    return x_train, x_test, y_train, y_test, classNames


# Performance Matrics
def performance_metrics(cnf_matrix, class_names):
    # Confusion Matrix Plot
    cmd = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
    cmd.plot(cmap='Greens')
    cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
    # Find All Parameters
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1-Score accuracy for each class
    FScore = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + TN + FN)
    print('\n\nClassName\tTP\tFP\tFN\tTN\tPrecision\tSensitivity\tSpecificity\tF-Score\t\tAccuracy')
    for i in range(len(class_names)):
        print(class_names[i] + "\t\t{0:.0f}".format(TP[i]) + "\t{0:.0f}".format(FP[i]) + "\t{0:.0f}".format(
            FN[i]) + "\t{0:.0f}".format(TN[i]) + "\t{0:.4f}".format(PPV[i]) + "\t\t{0:.4f}".format(
            TPR[i]) + "\t\t{0:.4f}".format(TNR[i]) + "\t\t{0:.4f}".format(FScore[i]) + "\t\t{0:.4f}".format(ACC[i]))

# Load data
x_train, x_test, y_train, y_test, classNames = loadDataset()
def unet_classification_model(input_shape, num_classes):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Middle
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Expanding path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    conv22 = Conv2D(512, 1, activation='relu', padding='same')(conv2)
    conv2_Max = MaxPooling2D(pool_size=(4, 4))(conv22)
    conv11 = Conv2D(512, 1, activation='relu', padding='same')(conv1)
    conv1_Max = MaxPooling2D(pool_size=(8, 8))(conv11)
    conv33 = Conv2D(512, 2, activation='relu', padding='same')(conv3)
    conv3_Max = MaxPooling2D(pool_size=(2, 2))(conv33)
    merge6 = concatenate([conv4, conv6, conv3_Max, conv2_Max, conv1_Max], axis=3)  # Concatenate conv1, conv2, conv3, conv4, and conv6
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    conv222 = Conv2D(256, 1, activation='relu', padding='same')(conv2)
    conv22_Max = MaxPooling2D(pool_size=(2,2 ))(conv222)
    conv111 = Conv2D(256, 1, activation='relu', padding='same')(conv2)
    conv111_Max = MaxPooling2D(pool_size=(2, 2))(conv111)
    merge7 = concatenate([conv3, conv7, conv22_Max, conv111_Max], axis=3)  # Concatenate conv1, conv2, conv3, and conv7
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    conv118 = Conv2D(128, 1, activation='relu', padding='same')(conv1)
    conv18_Max = MaxPooling2D(pool_size=(2, 2))(conv118)
    conv3_8 = Conv2D(128, 2, activation='relu', padding='same')(conv3)
    conv3_3_up = UpSampling2D(size=(2, 2))(conv3_8)
    merge8 = concatenate([conv2, conv8,conv18_Max,conv3_3_up], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = UpSampling2D(size=(2, 2))(conv7)
    conv6 = UpSampling2D(size=(4, 4))(conv6)
    conc_8_7_6=concatenate([conv7, conv8,conv6], axis=3)
    up9 = UpSampling2D(size=(2, 2))(conc_8_7_6)
    conv9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = concatenate([conv1, conv9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # Classification head
    flatten = Flatten()(conv9)
    dense1 = Dense(256)(flatten)
    leaky_relu = LeakyReLU(alpha=0.1)(dense1)  # You can adjust the alpha value as needed
    dropout = Dropout(0.5)(leaky_relu)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    return model

   # Initialize the model


unet_classification_Model = unet_classification_model((image_size, image_size, 3), len(classNames))

# Compile the model
unet_classification_Model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy',
                                  metrics=['accuracy'])
# Define callbacks
model_checkpoint = ModelCheckpoint('weights/unet_classification.hdf5', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1, verbose=1)
callbacks = [model_checkpoint, early_stopping, reduce_lr]
# Start Time
stTime = datetime.now()
# Train the model
history = unet_classification_Model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50,
                    batch_size=100, callbacks=callbacks)
# End Time
endTime = datetime.now()
# Visualize the model architecture
plot_model(unet_classification_Model, to_file='unet_classification_model.png', show_shapes=True, show_layer_names=True)
# Total Training Time
trainTime = endTime - stTime
print('Total Training Time : ', trainTime)
# predict the test Dataset
y_pred = unet_classification_Model.predict([x_test], batch_size=100)

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

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))

# Overall Performance 
performance_metrics(cm, classNames)   

# Evaluate and print test accuracy
score, acc = unet_classification_Model.evaluate(x_test, y_test, batch_size=100, verbose=1)
print("Test %s: %.2f%%" % (unet_classification_Model.metrics_names[1], acc * 100))

# Plot confusion matrix with green color
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)

# Set the color map to 'Greens'
disp.plot(cmap='Greens', values_format='d')

# Add a title to the plot
plt.title('Confusion Matrix')

# Save the figure
plt.savefig("confusion_matrix.png")

# Show the plot
plt.show()
