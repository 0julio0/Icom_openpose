#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:25:32 2020

@author: julioagonzalez
"""

import os
import cv2
import numpy as np
import glob
import json_lines
import random
from random import shuffle
import time

import paho.mqtt.client as mqtt
import json


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.publish(MQTT_TOPIC, 'Hello')

client = mqtt.Client()
client.on_connect = on_connect
# client.on_message = on_message
MQTT_TOPIC = 'send_mp4'
# client.connect("206.189.233.109", 1883, 60)
# client.connect("192.168.0.31", 1883, 60)
# client.connect("127.0.0.1", 1883, 60)
client.connect("ubuntu.local", 1883, 60)



word=[]
train_dataset = []


palavras = ['bomdia','porfavor','ligar','silencio']
img_height , img_width = 52, 52


vid = (glob.glob('output_json_icom/*'))
for arq in range(len(vid)):  
    with open(vid[arq], 'rb') as f: # opening file in binary(rb) mode  
        for item in json_lines.reader(f):
            matrix = item['frame']
            word.append(matrix)   
    train_dataset.append(word)
    word=[]


###exibe todos os videos armazenados 
for jj in range(len(train_dataset)):
    
    train_dataset2 = np.array(train_dataset[jj])
    for ss in range(len(train_dataset2)):
        start_time = time.time()
        ttt2 = train_dataset2[ss][:,:]
        #ttt2.shape
        
        
        image = ttt2.astype(np.uint8)
        # image = cv2.resize(image, (320, 240))
        # cv2.namedWindow("ICOM")                                
        # cv2.moveWindow("ICOM", 0,30)
        time.sleep(0.09)
        Final2 = image.tolist() 
        json_str = json.dumps({"class": "stream","frame":Final2})
        client.publish(MQTT_TOPIC, json_str,qos=0)

        cv2.imshow("ICOM",image)
        fps =  int(1.0 / (time.time() - start_time))
        print(image.shape, fps)
        if cv2.waitKey(1) == 2:                                
            break
    # image = np.zeros((240,320,3), np.uint8)
    # cv2.imshow("ICOM",image)
    # if cv2.waitKey(1) == 2:                                
    #     break
    # time.sleep(2)






