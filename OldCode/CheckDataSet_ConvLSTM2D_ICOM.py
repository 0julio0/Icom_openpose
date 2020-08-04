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

import time


word=[]
train_dataset = []


palavras = ['bomdia','porfavor','ligar','silencio']
img_height , img_width = 80, 80

for i in range(len(palavras)):
    vid = sorted(glob.glob('output_json_icom/'+palavras[i]+'*')) #busca as pastas com a palavra escolhida
    for arq in range(len(vid)):  
        with open(vid[arq], 'rb') as f: # opening file in binary(rb) mode  
            for item in json_lines.reader(f):
                matrix = item['frame']
                word.append(matrix)   
        train_dataset.append(word)
        word=[]

###print numeros de frames por video
# for jj in range(len(train_dataset)):
#     print(len(train_dataset[jj]))

###exibe todos os videos armazenados 
for jj in range(len(train_dataset)):
    train_dataset2 = np.array(train_dataset[jj])
    for ss in range(len(train_dataset2)):
        ttt2 = train_dataset2[ss][:,:]
        #ttt2.shape
        image = ttt2.astype(np.uint8)
        image = cv2.resize(image, (320, 240))
        cv2.namedWindow("ICOM")                                
        cv2.moveWindow("ICOM", 0,30)
        time.sleep(0.1)
        cv2.imshow("ICOM",image)
        if cv2.waitKey(1) == 2:                                
            break
    image = np.zeros((240,320,3), np.uint8)
    cv2.imshow("ICOM",image)
    if cv2.waitKey(1) == 2:                                
        break
    #time.sleep(2)






