"""
Recebe os pontos via mqtt e renderiza os pontos via opencv
"""
import os
import time
import numpy as np
import json
import cv2
import paho.mqtt.client as mqtt


MQTT_TOPIC = 'send_mp4'

frames=[]
start_time=0
BROKER = "ubuntu.local"
# BROKER = "206.189.233.109"
# BROKER = "127.0.0.1"
FRAME_W=320
FRAME_H=240

# FRAME_W=980
# FRAME_H=720

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(MQTT_TOPIC)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    frames.append(msg.payload) #cria uma fila com os pontos
  

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)

while True:
    start_time = time.time()
    client.loop()

    if(len(frames) > 0): #verifica se ha dados na lista
        
        data = json.loads(frames.pop()) 
        matrix = data['frame']
        ttt2 = np.array(matrix)
        image = ttt2.astype(np.uint8)
        # image = cv2.resize(image, (FRAME_W, FRAME_H))
        cv2.namedWindow("ICOM")                                
        cv2.imshow("ICOM",image)

        fps =  int(1.0 / (time.time() - start_time))
        print(ttt2.shape, fps)
        if cv2.waitKey(1) == 2:                                
            break


            
