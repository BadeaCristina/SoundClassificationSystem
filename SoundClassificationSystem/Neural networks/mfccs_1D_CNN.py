#%%
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers  import BatchNormalization
from keras import regularizers
# %%
#reading urbansound8k
def load_data(data_path, metadata_path):
    features = []
    labels = []

    metadata = pd.read_csv(metadata_path)
    for index, row in metadata.iterrows():
        file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

        print(file_path)
        target_sr = 22050
        audio, _ = librosa.load(file_path, sr=target_sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features.append(mfccs_scaled)
        labels.append(row['class'])


    return np.array(features), np.array(labels)
# %%
metadata_path = 'D:/1Disertatie/urbanDataset/UrbanSound8K.csv'
data_path = 'D:/1Disertatie/urbanDataset/ForFaceModel/'
features, labels = load_data(data_path, metadata_path)
######################################################################
# %% reading fma small dataset
data = pd.read_csv(os.path.join('D:/1Disertatie/fma_small/fma_metadata', 'tracks.csv'), index_col=0, header=[0, 1])
audio_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('D:/1Disertatie/fma_small/fma_small') for f in filenames if os.path.splitext(f)[1] == '.mp3']
features=[]
labels=[]
for audio_file in audio_files:
    track_id = int(os.path.splitext(os.path.basename(audio_file))[0])
    class_id = data.loc[track_id, ('track', 'genre_top')]   
    if pd.notna(class_id) :
        audio_file= audio_file.replace('\\','/')
        X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        target_sr = 22050
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=target_sr, n_mfcc=40).T,axis=0)
        features.append(mfccs) 
        labels.append(class_id)
features=np.array(features)
labels=np.array(labels)
#%%
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)
X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.1, random_state=42, stratify=labels_onehot)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
#%%
input_shape = (X_train.shape[1], 1)
m = Sequential()
m.add(Conv1D(128,
                input_shape=input_shape,
                kernel_size=3,
                strides=4,
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=2, strides=None))
m.add(Conv1D(128,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=2, strides=None))
m.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(MaxPooling1D(pool_size=2, strides=None))
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
m.add(Dense(8, activation='softmax'))
m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history=m.fit(X_train, 
          y_train,
          batch_size=128, 
          epochs=50,
          verbose=1,
          validation_data=(X_val, y_val))
# Evaluating the model on the training and testing set
score = m.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = m.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
# %%
