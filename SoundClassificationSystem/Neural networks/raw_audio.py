#%%
import os
import numpy as np
import pandas as pd
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from scipy.io import wavfile as wav
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense, Flatten
from tensorflow.keras.layers  import BatchNormalization
from sklearn import preprocessing
from keras.utils import to_categorical
from keras import regularizers
from keras.layers import Lambda
import keras.backend as K
#%%
import pandas as pd
data = pd.read_csv('D:/1Disertatie/urbanDataset/UrbanSound8K.csv')
#%%
TARGET_SR=8000
AUDIO_LENGTH=32000
def read_audio_from_filename(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    audio = audio.reshape(-1, 1)
    return audio
rawData=[];
labels=[];
for i in range(data.shape[0]):
    
    fullpath= os.path.join(os.path.abspath('D:/1Disertatie/urbanDataset/ForFaceModel/'),
                             'fold'+ str(data.fold[i])+'/',
                             str(data.slice_file_name[i]))
    class_id =data.classID[i]
    try:
        audio = read_audio_from_filename(fullpath, target_sr=TARGET_SR)
         # normalize mean 0, variance 1
        audio = (audio - np.mean(audio)) / np.std(audio)
        original_length = len(audio)
        if  original_length < AUDIO_LENGTH:
            audio=np.concatenate((audio, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            print('PAD New length =', len(audio))
        
        else:
            if  original_length > AUDIO_LENGTH:
                audio = audio[0:AUDIO_LENGTH]
                print('CUT New length =', len(audio))

        
    except Exception:
        print("Error encountered while parsing file: ", fullpath)

    rawData.append(audio)
    labels.append(class_id)

rawData= np.array(rawData)
labels= np.array(labels)
print(rawData.shape)
print(labels.shape)

#%%

le=preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded=le.transform(labels)
labels_encoded= to_categorical(labels_encoded, num_classes=10)

# %%
x_train,x_test,y_train,y_test = train_test_split(rawData, labels, test_size=0.1)
x_train,x_val,y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
m = Sequential()
m.add(Conv1D(128,
                input_shape=[AUDIO_LENGTH, 1],
                kernel_size=3,
                strides=4,
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))
m.add(Conv1D(128,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))
m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=4, strides=None))
m.add(Conv1D(512,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
m.add((Flatten()))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(Dense(512, activation='relu')) 
m.add(Dense(10, activation='softmax'))
m.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history2=m.fit(x_train, 
          y_train,
          batch_size=128, 
          epochs=50,
          verbose=1,
          validation_data=(x_val, y_val))
# Evaluating the model on the training and testing set
score = m.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = m.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])