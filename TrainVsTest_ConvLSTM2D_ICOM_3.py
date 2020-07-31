#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:25:32 2020

@author: julioagonzalez
"""
import tensorflow
#import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import json_lines
import random

import time
#import keras_metrics as km 

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
#from sklearn.metrics import multilabel_confusion_matrix

word=[]
label=[]
train_dataset = []
label_dataset = []

word_teste=[]
test_dataset = []
test_label_dataset = []
classes = ['Bomdia','PorFavor','Ligar','No_LIBRAS', 'eu', 'Quero', 'Obrigado', 'ajuda', 'medico','urgente','precisar']
# classes = ['bomdia','porfavor','ligar','silencio', 'eu', 'quero', 'obrigado', 'ajuda', 'medico','urgente','precisar']
# classes = ['bomdia','porfavor','ligar','silencio', 'eu', 'quero', 'obrigado']
# classes = ['bomdia','porfavor','silencio', 'eu', 'ligar', 'quero'   ]
img_height , img_width = 120, 120
seq_len = 18
total_len = 30
minimum_frame = 5
max_frame=total_len-seq_len-1
total_Loop = 15
loop2=1

img2 = np.zeros((img_height,img_width,1), np.uint8)

while loop2<=total_Loop:
    for i in range(len(classes)):
        vid = sorted(glob.glob('output_json_icom_webcam/'+classes[i]+'*')) #busca as pastas com a palavra escolhida
        for arq in range(len(vid)): 
            with open(vid[arq], 'rb') as f: # opening file in binary(rb) mode  
                pp=1
                first_frame = random.randint(minimum_frame, max_frame)
                for item in json_lines.reader(f):
                    
                    matrix = item['frame']
                    ttthr = np.array(matrix).reshape(-1,4)
                    Lista2 = ttthr.tolist()
                    for jj in range(len(Lista2)):
                        xa2=int(Lista2[jj][0]*img_width)
                        ya2=int(Lista2[jj][1]*img_height)
                        xb2=int(Lista2[jj][2]*img_width)
                        yb2=int(Lista2[jj][3]*img_height)
                        if (xa2 >0 and ya2 >0  and xb2 >0  and yb2 >0):
                            cv2.line(img2,(xa2, ya2),(xb2, yb2) , (255,255,255), 1)  
                    #if pp<=seq_len:   
                    #print(first_frame,pp,seq_len+first_frame)
                    if (pp >=first_frame and pp<(seq_len+first_frame)):    
                        word.append(img2)
                        img2 = np.zeros((img_height,img_width,1), np.uint8)
                    pp += 1
            y = [0]*len(classes)
            y[classes.index(classes[i])] = 1
            label_dataset.append(y)    
            train_dataset.append(word)
            word=[]
            label=[]
    loop2=loop2+1

    
#train_dataset = np.expand_dims(train_dataset, axis=4)
train_dataset = np.array(train_dataset)
label_dataset = np.array(label_dataset)

train_dataset.shape
X_train, X_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.20, shuffle=True, random_state=0)
# train_dataset.shape

model = Sequential()
model.add(ConvLSTM2D(filters = 32, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation = "softmax"))


# model = Sequential()
# model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 1)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(len(classes), activation = "softmax"))
 
model.summary()
 
opt = tensorflow.keras.optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
 
earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]

checkpoint_path = "training_ConvLSTM/icom_model_class11_web_seq18_120x120_5.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

#model.load_weights(checkpoint_path) 
history = model.fit(x = X_train, y = y_train, epochs=3, batch_size = 8 , shuffle=True, validation_split=0.2, callbacks=[cp_callback])

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.legend(loc='Loss')
# plt.show()

# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()



# ima = 9
# #X_test[ima].shape
# tt5 = np.expand_dims(X_test[ima], axis=0)
# #tt5.shape
# y_pred = model.predict(tt5) 
# y_pred = np.argmax(y_pred, axis = 1)
# print(classes[int(y_pred)])

# for ss in range(len(X_test[ima])):
#     ttt2 = X_test[ima][ss][:,:,-1]
#     #ttt2.shape
#     image = ttt2.astype(np.uint8)
#     image = cv2.resize(image, (360, 240))
#     cv2.namedWindow("ICOM")                                
#     cv2.moveWindow("ICOM", 0,30)
#     time.sleep(0.05)
#     cv2.imshow("ICOM",image)
#     if cv2.waitKey(1) == 2:                                
#         break






