"""
Recebe os pontos via mqtt e renderiza os pontos via opencv
"""
import json
import paho.mqtt.client as mqtt
import glob
import random
import json_lines
import json
from datetime import datetime
import tensorflow
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
import time
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from collections import Counter



MQTT_TOPIC = 'send_mp4'
BROKER = "192.168.0.11"
# BROKER = "192.168.0.15"
# BROKER = "127.0.0.1"
# BROKER = "ubuntu.local"

frames = []


# classes = ['bomdia','porfavor','ligar','silencio', 'eu', 'quero', 'obrigado']
classes = ['bomdia','porfavor','ligar','silencio', 'eu', 'quero', 'obrigado', 'ajuda', 'medico','urgente','precisar']



# img_height , img_width = 80, 80
img_height , img_width = 120, 120

seq_len = 18
texto=''

# # font 
# font = cv2.FONT_HERSHEY_SIMPLEX 

# # org 
# org = (50, 50) 
  
# # fontScale 
# fontScale = 1
   
# # Blue color in BGR 
# color = (255, 255, 255) 
  
# # Line thickness of 2 px 
# thickness = 2
# pool = ThreadPool(4)

frameviewTotal_array=[]

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(MQTT_TOPIC)
    # client.publish(MQTT_TOPIC, 'Hello')

def on_message(client, userdata, msg):
    frames.append(msg.payload) #cria uma fila com os pontos




# checkpoint_path = "training_ConvLSTM/icom_model_class7_seq18_100x100_1.hdf5"
# checkpoint_path = "training_ConvLSTM/icom_model_class7_seq18_110x110_2.hdf5"
# checkpoint_path = "training_ConvLSTM/icom_model_class7_seq18_120x120_3.hdf5"
checkpoint_path = "training_ConvLSTM/icom_model_class11_seq18_120x120_4.hdf5"

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
model.load_weights(checkpoint_path) 

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)

frameviewTotal=[]
Lista_Predict =[]


def prediction(img3):
    y_pred3 = model.predict(frameviewTotal_array) 
    y_pred3 = np.argmax(y_pred3, axis = 1)
    return y_pred3

img2 = np.zeros((img_height,img_width,1), np.uint8)
frames = []
while True:
    Lista2 = []
    


    client.loop()

    if(len(frames) > 0): #verifica se ha dados na lista
        start_time = time.time()
        data = json.loads(frames.pop()) #consome os pontos da fila criada
        try:
            frameview = data['frame']
            ttthr = np.array(frameview).reshape(-1,4)
            Lista2 = ttthr.tolist()
            for jj in range(len(Lista2)):

                xa2=int(Lista2[jj][0]*img_width)
                ya2=int(Lista2[jj][1]*img_height)
                xb2=int(Lista2[jj][2]*img_width)
                yb2=int(Lista2[jj][3]*img_height)

                if (xa2 >0 and ya2 >0  and xb2 >0  and yb2 >0):

                    cv2.line(img2,(xa2, ya2),(xb2, yb2) , (255,255,255), 1)    
        except:
            pass
        #frameviewTotal.append(img2)

        if len(frameviewTotal)==seq_len:
            frameviewTotal.remove(frameviewTotal[0]) 
        elif len(frameviewTotal)<seq_len:
            frameviewTotal.append(img2)
            if len(frameviewTotal)==seq_len:
                frameviewTotal_array = np.array(frameviewTotal)   
                frameviewTotal_array = np.expand_dims(frameviewTotal_array, axis=0)
                # y_pred = prediction(frameviewTotal_array)
                y_pred = model.predict(frameviewTotal_array) 
                y_pred = np.argmax(y_pred, axis = 1)

                Lista_Predict.append(classes[int(y_pred)])
                c = Counter(Lista_Predict)
                if len(Lista_Predict)==11:
                    Lista_Predict.remove(Lista_Predict[0])    
                select = 7
                
                for yy in range(len(classes)):
                    if c[classes[yy]]>select:
                        texto = str(classes[yy])    
                        if texto=='silencio':
                            texto=''
                fps =  int(1.0 / (time.time() - start_time))
                # print(classes[int(y_pred)], fps, c)
                print(texto,"   ", fps, c)
                # json_str = json.dumps({"predict":texto})
                # client.publish(MQTT_TOPIC, json_str,qos=0)



        img2 = np.zeros((img_height,img_width,1), np.uint8)

        


