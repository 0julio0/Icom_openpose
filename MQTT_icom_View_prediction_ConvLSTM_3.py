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
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
import threading
import re

MQTT_TOPIC = 'send_mp4'
# BROKER = "192.168.0.16"
# BROKER = "192.168.0.15"
# BROKER = "127.0.0.1"
BROKER = "ubuntu.local"

frames = []

start_time=0


img_w = 720
img_h = 405

str1=''

seq_len = 18

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
texto=''
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 255, 255) 
  
# Line thickness of 2 px 
thickness = 2
pool = ThreadPool(4)
frase = []
frameviewTotal_array=[]

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    frames.append(msg.payload) #cria uma fila com os pontos




client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)

frameviewTotal=[]
Lista_Predict =[]

img = np.zeros((img_h,img_w,3), np.uint8)
frames = []
while True:
    Lista2 = []
    
    # img = np.zeros((img_h,img_w,3), np.uint8)
    client.loop()

    if(len(frames) > 0): #verifica se ha dados na lista
        # start_time = time.time()
        data = json.loads(frames.pop()) #consome os pontos da fila criada
        try:
            frameview = data['frame']
            ttthr = np.array(frameview).reshape(-1,4)
            Lista2 = ttthr.tolist()
            for jj in range(len(Lista2)):
                xa=int(Lista2[jj][0]*img_w)
                ya=int(Lista2[jj][1]*img_h)
                xb=int(Lista2[jj][2]*img_w)
                yb=int(Lista2[jj][3]*img_h)



                if (xa >0 and ya >0  and xb >0  and yb >0):
                    cv2.line(img,(xa, ya),(xb, yb) , (255,255,255), 2) 



            cv2.imshow("ICOM",img)
            img = np.zeros((img_h,img_w,3), np.uint8)
            if cv2.waitKey(1) == 2:                                
                break
 
        except:
            frameview2 = data['predict']
            # print(frameview2)
            # ttmm =(time.time()-start_time)
            if frameview2!='':
                # c = Counter(frase)
                if frameview2 not in frase:
                    frase.append(frameview2)
                    start_time = time.time()
                    str1 = ' '.join(frase)
                    print(str1)
            elif len(frase)>5 or (time.time()-start_time)>5:
                frase=[]
                str1 = ' '.join(frase)                
            cv2.putText(img, str1, (20,30), font, fontScale, color, 1, cv2.LINE_AA)
     



