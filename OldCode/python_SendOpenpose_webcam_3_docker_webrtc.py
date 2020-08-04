# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
import paho.mqtt.client as mqtt
import json

import glob
import random
import json_lines
from datetime import datetime
import tensorflow
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from collections import Counter

import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()


frame_w2 = 120
frame_h2 = 120

img_height , img_width = 120, 120



POSEBody_PAIRS = [ [1,5],[2,1],[5,6],[2,3],[6,7],[3,4],\
    [25,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,36],[36,37],[37,38],[38,39],[39,40],[40,41],\
        [42,43],[43,44],[44,45],[45,46],\
        [47,48],[48,49],[49,50],[50,51],\
        [52,53],[53,54],[54,55],\
          [56,57],[57,58],[58,59] ,[59,60],\
          [61,62],[62,63],[63,64],[64,65],[65,66],[66,61],\
           [67,68],[68,69],[69,70],[70,71],[71,72],[72,67],\
            [73,74],[74,75],[75,76],[76,77],[77,78],[78,79],[79,80],[80,81],[81,82],[82,83],[83,84],[84,73],\
               [85,86],[86,87],[87,88],[88,89],[89,90],[90,91],[91,92],[92,85],\
                 [95,96],[96,97],[97,98],[98,99],\
                     [95,100],[100,101],[101,102],[102,103],\
                         [95,104],[104,105],[105,106],[106,107],\
                             [95,108],[108,109],[109,110],[110,111],\
                                 [95,112],[112,113],[113,114],[114,115],\
                                     [116,117],[117,118],[118,119],[119,120],\
                                         [116,121],[121,122],[122,123],[123,124],\
                                           [116,125],[125,126],[126,127],[127,128],\
                                               [116,129],[129,130],[130,131],[131,132],\
                                                 [116,133],[133,134],[134,135],[135,136]    ]

olho = [93, 94]
img = np.zeros((frame_h2,frame_w2,3), np.uint8) #Configura matriz do fundo da imagem e resolucao

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.publish(MQTT_TOPIC, 'Hello')

# def on_message(client, userdata, msg):
#     frames.append(msg.payload) #cria uma fila com os pontos

client = mqtt.Client()
client.on_connect = on_connect

MQTT_TOPIC = 'send_mp4'
# client.connect("206.189.233.109", 1883, 60)
client.connect("192.168.0.12", 1883, 60)



poseBody = []
render_threshold = 0.01


texto=''

loc =0
posic0x = .5
posic0y = .6

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append('../../python');
# sys.path.append('/usr/local/python');
from openpose import pyopenpose as op
# Flags
parser = argparse.ArgumentParser()
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["num_gpu"] = 1
# params["net_resolution"] = "128x128"
# params["net_resolution"] = "240x240"
params["net_resolution"] = "272x272"
# params["net_resolution"] = "-1x240"
# params["net_resolution"] = "320x320"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
# params["render_threshold"] = 0.05
params["render_pose"] = 0
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["tracking"] = 1
params["number_people_max"] = 1
params["disable_blending"] = True
params["face"] = True
params["hand"] = True
params["frame_flip"] = True

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

frameviewTotal=[]
Lista_Predict =[]

frames = []

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """
    kind = "video"

    def __init__(self, track):  
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        start_time2 = time.time()
        global loc
        global posic0x
        global posic0y
        global difx0
        global dify0
        global difx
        global dify
        global start_time
        corpo = []
        cara = []
        hand_right =[]
        hand_left = []
        Lista = []
        Lista2 = []
        Final = []
        global img2
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img,180) 
        # print(img.shape)
        tm = (time.time() - start_time)
        if tm>=0.0599:
            datum = op.Datum()
            imageToProcess = (img)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            fps =  (1.0/(time.time() - start_time))
            FRAME_HEIGHT, FRAME_WIDTH, colors = img.shape
            img2 = np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3), np.uint8)
            try:
                ttthr = np.array(datum.poseKeypoints[0])
                if loc==0:
                    posicx = ttthr[1][0]/FRAME_WIDTH
                    posicy = ttthr[1][1]/FRAME_HEIGHT
                    difx =  int((posicx - posic0x)*FRAME_WIDTH)
                    dify =  int((posicy - posic0y)*FRAME_HEIGHT)
                    difx0 =  (posicx - posic0x)
                    dify0 =  (posicy - posic0y)
                    loc=1
                for hh in range(len(ttthr)):
                    ttthr[hh][0]=ttthr[hh][0]/FRAME_WIDTH
                    ttthr[hh][1]=ttthr[hh][1]/FRAME_HEIGHT
                    if ttthr[hh][2]<render_threshold:
                        ttthr[hh][0]=0
                        ttthr[hh][1]=0
                pose_keypoints_2d2 = ttthr.tolist()
                for handR in range(len(pose_keypoints_2d2)):
                    corpo.append(pose_keypoints_2d2[handR][0])
                    corpo.append(pose_keypoints_2d2[handR][1])
                    corpo.append(pose_keypoints_2d2[handR][2])
                pose_keypoints_2d = corpo

                ttthr = np.array(datum.faceKeypoints[0])
                for hh in range(len(ttthr)):
                    ttthr[hh][0] = ttthr[hh][0]/FRAME_WIDTH
                    ttthr[hh][1] = ttthr[hh][1]/FRAME_HEIGHT                   
                    if ttthr[hh][2]<render_threshold:
                        ttthr[hh][0]=0
                        ttthr[hh][1]=0
                face_keypoints_2d2 = ttthr.tolist()
                for handR in range(len(face_keypoints_2d2)):
                    cara.append(face_keypoints_2d2[handR][0])
                    cara.append(face_keypoints_2d2[handR][1])
                    cara.append(face_keypoints_2d2[handR][2])
                face_keypoints_2d = cara
            
                ttthr = np.array(datum.handKeypoints[0][0])
                for hh in range(len(ttthr)):
                    ttthr[hh][0] = ttthr[hh][0]/FRAME_WIDTH
                    ttthr[hh][1] = ttthr[hh][1]/FRAME_HEIGHT                               
                    if ttthr[hh][2]<render_threshold:
                        ttthr[hh][0]=0
                        ttthr[hh][1]=0
                hand_right_keypoints_2d2 = ttthr.tolist()
                for handR in range(len(hand_right_keypoints_2d2)):
                    hand_right.append(hand_right_keypoints_2d2[handR][0])
                    hand_right.append(hand_right_keypoints_2d2[handR][1])
                    hand_right.append(hand_right_keypoints_2d2[handR][2])
                hand_right_keypoints_2d = hand_right

                ttthr = np.array(datum.handKeypoints[1][0])
                for hh in range(len(ttthr)):
                    ttthr[hh][0] = ttthr[hh][0]/FRAME_WIDTH
                    ttthr[hh][1] = ttthr[hh][1]/FRAME_HEIGHT                      
                    if ttthr[hh][2]<render_threshold:
                        ttthr[hh][0]=0
                        ttthr[hh][1]=0
                hand_left_keypoints_2d2 = ttthr.tolist()
                for handR in range(len(hand_left_keypoints_2d2)):
                    hand_left.append(hand_left_keypoints_2d2[handR][0])
                    hand_left.append(hand_left_keypoints_2d2[handR][1])
                    hand_left.append(hand_left_keypoints_2d2[handR][2])
                hand_left_keypoints_2d = hand_left



                for nee in range(len(pose_keypoints_2d)):
                    prod = pose_keypoints_2d[nee]
                    Lista.append(prod)
                for nee in range(len(face_keypoints_2d)):
                    prod2 = face_keypoints_2d[nee]
                    Lista.append(prod2) 
                for nee in range(len(hand_right_keypoints_2d)):
                    prod3 = hand_right_keypoints_2d[nee]
                    Lista.append(prod3)
                for nee in range(len(hand_left_keypoints_2d)):
                    prod4 = hand_left_keypoints_2d[nee]
                    Lista.append(prod4)       


                ttt = np.array(Lista).reshape(-1,3)  
                ttt2 = ttt[:,:-1]
                Lista2 = ttt2.tolist()
                for pairb in POSEBody_PAIRS:
                    partAb = pairb[0] #identifica do ponto
                    partBb = pairb[1]

                    xa2=int((Lista2[partAb][0]*FRAME_WIDTH))-difx
                    ya2=int((Lista2[partAb][1]*FRAME_HEIGHT))-dify
                    xb2=int((Lista2[partBb][0]*FRAME_WIDTH))-difx
                    yb2=int((Lista2[partBb][1]*FRAME_HEIGHT))-dify  

                    xa=Lista2[partAb][0]-difx0
                    ya=Lista2[partAb][1]-dify0
                    xb=Lista2[partBb][0]-difx0
                    yb=Lista2[partBb][1]-dify0
                    
                    if (xa>0 and ya >0  and xb >0  and yb>0):  
                        cv2.line(img2,(xa2, ya2),(xb2, yb2) , (255,255,255), 2)   
                        Final.append(xa)
                        Final.append(ya)
                        Final.append(xb)
                        Final.append(yb)
                # tm = (time.time() - start_time)
                ttt_final = np.array(Final).reshape(-1,2)
                Final2 = ttt_final.tolist() 
                json_str = json.dumps({"class": "stream","frame":Final2})
                client.publish(MQTT_TOPIC, json_str,qos=0)
                try:
                    fps =  (1.0/(time.time() - start_time2))  
                    print(int(fps), img.shape)
                except:
                    pass    
                start_time = time.time()
            except:
                    pass  

        
        new_frame = VideoFrame.from_ndarray(img2, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
        # return frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            # recorder.addTrack(track)
            pass
        elif track.kind == "video":
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            # await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    # await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    start_time = time.time()
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)