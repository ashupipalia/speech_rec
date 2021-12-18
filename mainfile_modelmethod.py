import os
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
# import wavfile
# filepath='/Users/lakshaykalra/Desktop/Speech recognisation/data/Bye/Bye0.wav'
train_audio_path='/Users/lakshaykalra/Desktop/Speech recognisation/data'
# def create_model(train_audio_path):

# def create_model(train_audio_path):
no_of_recordings=[]

labels=os.listdir(train_audio_path)
# print(labels)
labels=labels[1:]

for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
#plot
# plt.figure(figsize=(30,5))
# index = np.arange(len(labels))
# plt.bar(index, no_of_recordings)
# plt.xlabel('Commands', fontsize=12)
# plt.ylabel('No of recordings', fontsize=12)
# plt.xticks(index, labels, fontsize=15, rotation=60)
# plt.title('No. of recordings for each command')
# plt.show()

all_wave = []
all_label = []
for label in labels:
    # print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 44100)
        samples = librosa.resample(samples, sample_rate, 8000)
        # print(len(samples))
        if(len(samples)== 24000) : 
            all_wave.append(samples)
            all_label.append(label)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

import tensorflow as tf

y=tf.keras.utils.to_categorical(y, num_classes=len(labels), dtype='float32')
all_wave = np.array(all_wave).reshape(-1,24000,1)
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)


inputs = tf.keras.layers.Input(shape=(24000,1))

#First Conv1D layer
conv = tf.keras.layers.Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = tf.keras.layers.MaxPooling1D(3)(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

#Second Conv1D layer
conv = tf.keras.layers.Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = tf.keras.layers.MaxPooling1D(3)(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

#Third Conv1D layer
conv = tf.keras.layers.Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = tf.keras.layers.MaxPooling1D(3)(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

#Fourth Conv1D layer
conv = tf.keras.layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = tf.keras.layers.MaxPooling1D(3)(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

#Flatten layer
conv = tf.keras.layers.Flatten()(conv)

#Dense Layer 1
conv = tf.keras.layers.Dense(256, activation='relu')(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

#Dense Layer 2
conv = tf.keras.layers.Dense(128, activation='relu')(conv)
conv = tf.keras.layers.Dropout(0.2)(conv)

outputs = tf.keras.layers.Dense(len(labels), activation='softmax')(conv)

model = tf.keras.models.Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.00001) 
# mc = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

   
# history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=2, validation_data=(x_val,y_val))
history=model.fit(x_tr, y_tr ,epochs=100, batch_size=2, validation_data=(x_val,y_val))

# from matplotlib import pyplot 
# pyplot.plot(history.history['loss'], label='train') 
# pyplot.plot(history.history['val_loss'], label='test') 
# pyplot.legend()
# pyplot.show()




def predict(audio):
    prob=model.predict(audio.reshape(1,24000,1))
    index=np.argmax(prob[0])
    return classes[index]

# import random
# index=random.randint(0,len(x_val)-1)
# samples=x_val[index].ravel()
# print("Audio:",classes[np.argmax(y_val[index])])
# ipd.Audio(samples, rate=24000)
audio_file='/Users/lakshaykalra/Desktop/Speech recognisation/recording1.wav'
# sample_rate, samples = wavfile.read(audio_file)
samples,sample_rate = librosa.load(audio_file,sr=44100)
samples=np.array(samples,dtype='float64')
samples = librosa.resample(samples, sample_rate, 8000)

samples=samples.ravel()
print("Text:",predict(samples))

