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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import keras_efficientnet_v2
from tensorflow.keras.layers.experimental import preprocessing
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from datetime import datetime
import warnings
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
import cv2
import random

warnings.filterwarnings("ignore")

# Random Seed Fix output
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
dirPath = 'uniFeature'

# Image Size
image_size = 32

# Split Ratio
test_ratio = 0.2
dropout_rate = 0.2
no_epochs = 50
num_class = 21
batchSize = 100


# Normalization of Data
def NormalizeData(data):
    if (np.max(data) - np.min(data)) != 0:
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif np.max(data) != 0:
        return data / np.max(data)
    else:
        return data


# Loading the Dataset
def loadFeaturedDataset():
    # the data, shuffled and split between train and test sets
    imgArr = []
    image_label = []
    class_names = []
    dirList = [f for f in listdir(pathImg) if not isfile(join(pathImg, f))]
    if not os.path.exists(dirPath):
        # Create a new directory because it does not exist
        os.makedirs(dirPath)
    for dirName in dirList:
        classDirPath = os.path.join(dirPath, dirName)
        if not os.path.exists(classDirPath):
            os.makedirs(classDirPath)
    for i in range(len(dirList)):
        fileList = list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg + '/' + dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        print(dirList[i], len(fileList))
        for j in range(len(fileList)):
            if (fileList[j].endswith('.jpg')):
                try:
                    imgLoad = Image.open(fileList[j])
                    numImg = (np.array(imgLoad)).astype('float64')
                    image_label.append(i)
                    class_names.append(dirList[i])
                    img = cv2.cvtColor(cv2.imread(fileList[j]), cv2.COLOR_BGR2GRAY)
                    imgArr.append(img)
                except:
                    print('Problem in Reading : ', fileList[j])
                    # List to numpy Array and Reshape
    imgArr = np.array(imgArr)
    image_label = np.array(image_label)
    imgArr = imgArr.reshape((len(imgArr), -1))
    print(imgArr.shape)

    # Batchwise Mean values
    mnImgArr = []
    mnLblArr = []
    batch = int(len(imgArr) / batchSize)
    for i in range(batch):
        mnImgArr.append(np.mean(imgArr[i * batchSize:((i + 1) * batchSize) - 1], axis=0))
        mnLblArr.append(np.mean(image_label[i * batchSize:((i + 1) * batchSize) - 1]))
    mnImgArr = np.array(mnImgArr)
    mnLblArr = np.array(mnLblArr)

    # Feature Selection
    totalFeature = 16 * 16
    # Create the Univariate object and rank each pixel of 25% of Data
    selector = SelectKBest(f_classif, k=totalFeature)
    selector.fit(mnImgArr, mnLblArr)
    featureArr = selector.transform(mnImgArr)

    # Feature Selection on Image Data
    featureSelect = selector.get_support()
    featureSelectF = np.zeros((image_size * image_size,), dtype=bool)
    counter = 0
    for i in range(len(featureSelect)):
        if counter < totalFeature and featureSelect[i] == True:
            featureSelectF[i] = featureSelect[i]
            counter = counter + 1
    maskFeatureSelect = featureSelectF.astype(np.uint8).reshape(image_size, image_size)
    maskFeatureSelectRGB = ~featureSelectF
    maskFeatureSelectRGB = maskFeatureSelectRGB.astype(np.uint8).reshape(-1)
    maskFeatureSelectRGB = np.repeat(maskFeatureSelectRGB, 3)
    print('Featured Select Shape : ', maskFeatureSelect.shape)

    fsimgArr = []
    fslabelArr = []
    # Save Featured Data
    for i in range(len(dirList)):
        fileList = list()
        for (dirpath, dirnames, filenames) in os.walk(pathImg + '/' + dirList[i]):
            fileList += [os.path.join(dirpath, file) for file in filenames]
        for j in range(len(fileList)):
            if (fileList[j].endswith('.jpg')):
                try:
                    # Featured Image save
                    img = cv2.imread(fileList[j])
                    fsImg = cv2.bitwise_and(img, img, mask=maskFeatureSelect)
                    fsFile = fileList[j].replace(pathImg, dirPath)
                    cv2.imwrite(fsFile, fsImg)
                    # Featured Image as 25% of whole image 32x32
                    imgFS = np.ma.array(fsImg.reshape(-1), mask=maskFeatureSelectRGB)
                    imgFS = np.ma.MaskedArray.compressed(imgFS)
                    imgFS = ~imgFS
                    normImg = NormalizeData(imgFS.astype('float64')) * ((i + 1) / len(dirList))
                    fsimgArr.append(normImg)
                    fslabelArr.append(i)
                except:
                    print('Problem in File : ', fileList[j])

    fsimgArr = np.array(fsimgArr)
    classNames = sorted(set(class_names), key=class_names.index)
    fslabArr = to_categorical(np.array(fslabelArr))
    # Fix stratified sampling split
    x_train, x_test, y_train, y_test = train_test_split(fsimgArr, fslabArr, test_size=test_ratio, random_state=SEED,
                                                        stratify=fslabArr)
    xtrain, xtest, ytrain, ytest = train_test_split(fsimgArr, fslabArr, test_size=1 - test_ratio, random_state=SEED,
                                                    stratify=fslabArr)

    # Reshape the data as model is 32x32
    X_train = x_train.reshape(-1, image_size, image_size, 3)
    X_test = xtest.reshape(-1, image_size, image_size, 3)
    Y_train = []
    for i in range(len(y_train)):
        if i % 4 == 0: Y_train.append(y_train[i])
    Y_train = np.array(Y_train)
    Y_test = []
    for i in range(len(ytest)):
        if i % 4 == 0: Y_test.append(ytest[i])
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test, classNames


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


# load data
x_train, x_test, y_train, y_test, classNames = loadFeaturedDataset()


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
    merge6 = concatenate([conv4, conv6, conv3_Max, conv2_Max, conv1_Max],
                         axis=3)  # Concatenate conv1, conv2, conv3, conv4, and conv6
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    conv222 = Conv2D(256, 1, activation='relu', padding='same')(conv2)
    conv22_Max = MaxPooling2D(pool_size=(2, 2))(conv222)
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
    merge8 = concatenate([conv2, conv8, conv18_Max, conv3_3_up], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = UpSampling2D(size=(2, 2))(conv7)
    conv6 = UpSampling2D(size=(4, 4))(conv6)
    conc_8_7_6 = concatenate([conv7, conv8, conv6], axis=3)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

# ...

# Number of folds for cross-validation
num_folds = 5 # You can adjust this number as needed

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

# Store performance metrics for each fold
all_accuracies = []
all_confusion_matrices = []
all_test_predictions = []  # To store predictions on the test set

# Perform k-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(skf.split(x_train, np.argmax(y_train, axis=1))):
    print(f"\nTraining Fold {fold + 1}/{num_folds}")

    x_train_fold, x_val_fold = x_train[train_indices], x_train[val_indices]
    y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]
    tf.keras.backend.clear_session()

    # Initialize the model
    unet_classification_Model = unet_classification_model((image_size, image_size, 3), len(classNames))

    # Compile the model
    unet_classification_Model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy',
                                      metrics=['accuracy'])

    # Train the model
    history = unet_classification_Model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold),
                                            epochs=30, batch_size=100, callbacks=callbacks, verbose=1)

    # Evaluate on the validation set
    y_val_pred = unet_classification_Model.predict(x_val_fold, batch_size=100)
    val_cm = confusion_matrix(np.argmax(y_val_fold, 1), np.argmax(y_val_pred, 1))
    val_acc = accuracy_score(np.argmax(y_val_fold, 1), np.argmax(y_val_pred, 1))
    print(f"\nValidation Accuracy - Fold {fold + 1}: {val_acc * 100:.2f}%")

    # Store performance metrics for this fold
    all_accuracies.append(val_acc)
    all_confusion_matrices.append(val_cm)

    # Evaluate on the test set and store predictions for each fold
    test_predictions = unet_classification_Model.predict(x_test, batch_size=32)
    all_test_predictions.append(test_predictions)

# Compute the overall test accuracy using the accumulated predictions
all_test_predictions = np.mean(all_test_predictions, axis=0)
overall_test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(all_test_predictions, axis=1))
print(f"\nOverall Test Accuracy Across {num_folds} Folds: {overall_test_accuracy * 100:.2f}%")


# Display overall performance metrics using the average confusion matrix across all folds
overall_cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(all_test_predictions, axis=1))
print(f"\nOverall Confusion Matrix:")
print(overall_cm)
performance_metrics(overall_cm, classNames)

# Draw accuracy and loss curves for the overall model
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy - Overall')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss - Overall')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()