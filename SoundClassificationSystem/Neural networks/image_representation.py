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
# %%
import pandas as pd
data = pd.read_csv('D:/1Disertatie/urbanDataset/UrbanSound8K.csv')
#%%
print(data.shape[0])
dataset = np.zeros(shape = (data.shape[0],2),dtype = object)
dataset.shape
#%%
print(data.slice_file_name[0])
print(data.classID[0])
#%%
N_FFT = 512
HOP_LENGTH = int(N_FFT / 2)
### Define helper functions ###
bar = progressbar.ProgressBar(maxval=data.shape[0], widgets=[progressbar.Bar('$', '||', '||'), ' ', progressbar.Percentage()])
bar.start()
for i in range(data.shape[0]):
    
    fullpath= os.path.join(os.path.abspath('D:/1Disertatie/urbanDataset/ForFaceModel/'),
                             'fold'+ str(data.fold[i])+'/',
                             str(data.slice_file_name[i]))
    class_id =data.classID[i]
    try:
        X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=X, sr=sample_rate).T,axis=0)
        chroma_cqt = np.mean(librosa.feature.chroma_cqt(y=X, sr=sample_rate).T,axis=0)
        spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=180).T,axis=0)
    except Exception:
        print("Error encountered while parsing file: ", fullpath)
        spectrogram,class_id = None, None
    feature = spectrogram
    label = class_id
    dataset[i,0],dataset[i,1] = feature,label
    
    bar.update(i+1)
# %%
np.save("dataset_chroma_cens_urban.npy",dataset,allow_pickle=True)
# %%
# %%
data = pd.DataFrame(np.load("dataset_chroma_cens_urban.npy",allow_pickle= True))
data.columns = ['feature', 'label']
X = np.array(data.feature.tolist())
y = np.array(data.label.tolist())

lb = LabelEncoder()
labels = np_utils.to_categorical(lb.fit_transform(y))
#%%
print(X.shape)
#%%
X_train = np.array([x.reshape( (32, 16, 3) ) for x in X])

# %%
X_train,x_test,y_train,y_test = train_test_split(X_train, labels, test_size=0.2, random_state = 42)
X_train,x_val,y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state = 42)
# %%
print(X_train.shape)
print(x_test.shape)
print(x_val.shape)
# %%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

input_shape = (32, 16, 3)
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D,Dropout, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
model2 = Sequential()

model2.add(Conv2D(64,kernel_size=(3,3),
          strides=(1,1), activation= 'relu',padding='same',input_shape=input_shape ))
model2.add(MaxPooling2D((2,2), padding='same'))

model2.add(Conv2D(128, kernel_size=(3,3), activation='relu',padding='same'))
model2.add(MaxPooling2D((2,2), padding='same'))
model2.add(Dropout(0.2))

model2.add(Conv2D(128, kernel_size=(3,3), activation='relu',padding='same'))
model2.add(MaxPooling2D((2,2), padding='same'))
model2.add(Dropout(0.2))

model2.add(Flatten())
 


model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.2))


model2.add(Dense(10, activation='softmax'))
opt1= Adam(learning_rate=0.0001)
model2.compile(optimizer=opt1,loss="categorical_crossentropy",metrics=["accuracy"])
# %%
history2 = model2.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(x_val, y_val))
# %%
score = model2.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model2.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
# %%
model2.save('CNN_chromaCENS_100Acc0.5787.h5')
# %%
print ("Prediction with [train] data")
y_pred = model2.predict_classes(x_test)
#%%
# show sample results
print ("---samples---")
contor=0
for i in range(1700):
    print (i,"predict =", y_pred[i])
    print (i,"original=", argmax(y_test[i]))
    print ("")
    if y_pred[i] == argmax(y_test[i]):
        contor=contor+1
# %%
print(contor)
