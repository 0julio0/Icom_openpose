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



#palavras = ['bomdia','porfavor','ligar','silencio']
palavras = ['bomdia','porfavor','silencio']
img_height , img_width = 120, 120

for i in range(len(palavras)):
    Lista2=[]
    img2 = np.zeros((img_height,img_width,1), np.uint8)
    vid = sorted(glob.glob('output_json_icom/'+palavras[i]+'*')) #busca as pastas com a palavra escolhida
    for arq in range(len(vid)):  
        with open(vid[arq], 'rb') as f: # opening file in binary(rb) mode  
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
                word.append(img2)  
                img2 = np.zeros((img_height,img_width,1), np.uint8)
        train_dataset.append(word)
        word=[]




#print numeros de frames por video
 for jj in range(len(train_dataset)): 
     print(len(train_dataset[jj]))

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






