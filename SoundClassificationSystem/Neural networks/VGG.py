#%%
import time
import progressbar
import os
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa # for sound processing
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from numpy import argmax
import cv2
from PIL import Image
import pandas as pd
#%%
data = pd.read_csv('D:/1Disertatie/urbanDataset/UrbanSound8K.csv')
images=[];
labels=[]

for i in range(data.shape[0]):
    
    fullpath= os.path.join(os.path.abspath('D:/1Disertatie/urbanDataset/ForFaceModel/'),
                             'fold'+ str(data.fold[i])+'/',
                             str(data.slice_file_name[i]))
    class_id =data.classID[i]
    try:
        X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
        spectogram = librosa.feature.melspectrogram(y=X, sr=sample_rate,n_mels=128)
        spectogram = librosa.power_to_db(spectogram,ref=np.max)
        spectrogram = (255 * (spectogram - np.min(spectogram)) / np.ptp(spectogram)).astype(np.uint8)
        spectogram = cv2.resize(spectogram, (256,256)) 
        spectrogram_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        spectrogram_rgb[:, :, 0] = spectogram
        spectrogram_rgb[:, :, 1] = spectogram
        spectrogram_rgb[:, :, 2] = spectogram
    except Exception:
        print("Error encountered while parsing file: ", fullpath)
        image_rgb,label = None, None
    images.append(spectrogram_rgb)
    labels.append(class_id)
# %%
images= np.array(images)
labels= np.array(labels)

#%%
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded=le.transform(labels)

# %%
x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.1, random_state = 42)
x_train,x_val,y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state = 42)
# %%
y_train_one_hot= to_categorical(y_train)
y_test_one_hot= to_categorical(y_test)
y_val_one_hot= to_categorical(y_val)
# %%
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D,Dropout, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D)


VGG_model = VGG16(include_top=False, input_shape=(256,256,3),weights = 'imagenet')
model = Sequential()
model.add(VGG_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#%%

from tensorflow import keras 
activation='softmax'
n_classes=10
LR = 0.0001
optim = keras.optimizers.Adam(LR)
# %%
model.compile(optim, loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# %%
history=model.fit(x_train, 
          y_train_one_hot,
          batch_size=128, 
          epochs=4,
          verbose=1,
          validation_data=(x_val, y_val_one_hot))
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train_one_hot, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test_one_hot, verbose=0)
print("Testing Accuracy: ", score[1])
# %%
VGG_model = VGG16(include_top=False, input_shape=(256,256,3),weights = 'none')
pretrainedModel = Sequential()
pretrainedModel.add(VGG_model)
pretrainedModel.add(GlobalAveragePooling2D())
pretrainedModel.add(Dense(1024,activation='relu'))
pretrainedModel.add(Dropout(0.5))
pretrainedModel.add(Dense(10,activation='softmax'))
# %%
pretrainedModel.compile(optim, loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# %%
history2=pretrainedModel.fit(x_train, 
          y_train_one_hot,
          batch_size=128, 
          epochs=8,
          verbose=1,
          validation_data=(x_val, y_val_one_hot))
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Train ac')
plt.plot(epochs, val_acc, 'b', label='Validation ac')
plt.title('Validation and train accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.show()


