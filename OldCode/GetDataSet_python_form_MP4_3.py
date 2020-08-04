# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time
import json
import json_lines
from datetime import datetime
import glob
import random

frame_w2 = 480
frame_h2 = 270

# frame_w2 = 1280
# frame_h2 = 720

# FRAME_WIDTH=1920
# FRAME_HEIGHT=1080

FRAME_WIDTH=1280
FRAME_HEIGHT=720

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

palavra = str(sys.argv[1])



poseBody = []
render_threshold = 0.01

sys.path.append('../../python');
from openpose import pyopenpose as op
# Flags
parser = argparse.ArgumentParser()
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["num_gpu"] = 1
# params["net_resolution"] = "128x128"
params["net_resolution"] = "240x240"
# params["net_resolution"] = "320x320"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_pose"] = 0

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



def getFrame(sec):
    img = np.zeros((frame_h2,frame_w2,3), np.uint8)
    global loc
    global posic0x
    global posic0y
    global difx0
    global dify0
    global difx
    global dify    
    vid_capture.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid_capture.read()
    if hasFrames:
        corpo = []
        cara = []
        hand_right =[]
        hand_left = []
        Lista = []
        Lista2 = []
        imgList=[]
        imgFinal=[]
        Final = []

        ret,frame = vid_capture.read()
        frame = cv2.flip(frame,180) 
        datum = op.Datum()
        imageToProcess = (frame)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        ttthr = np.array(datum.poseKeypoints[0])
        if loc==0:
            posicx = ttthr[1][0]/FRAME_WIDTH
            posicy = ttthr[1][1]/FRAME_HEIGHT
            difx =  int((posicx - posic0x)*frame_w2)
            dify =  int((posicy - posic0y)*frame_h2)
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

            xa=Lista2[partAb][0]-difx0
            ya=Lista2[partAb][1]-dify0
            xb=Lista2[partBb][0]-difx0
            yb=Lista2[partBb][1]-dify0

            Final.append(xa)
            Final.append(ya)
            Final.append(xb)
            Final.append(yb)
    
            xa2=int((Lista2[partAb][0]*frame_w2))-difx
            ya2=int((Lista2[partAb][1]*frame_h2))-dify
            xb2=int((Lista2[partBb][0]*frame_w2))-difx
            yb2=int((Lista2[partBb][1]*frame_h2))-dify
            try:
                if (xa2 >0 and ya2 >0  and xb2 >0  and yb2 >0):
                    cv2.line(img,(xa2, ya2),(xb2, yb2) , (255,255,255), 1)    
            except:
                pass

        ttt_final = np.array(Final).reshape(-1,2)
        Final2 = ttt_final.tolist() 
        json_str = json.dumps({"class": palavra,"frame":Final2})
        file.write(json_str+ '\n')

        cv2.namedWindow("ICOM")
        cv2.imshow("ICOM",img)

        img = np.zeros((frame_h2,frame_w2,3), np.uint8) 
        k = cv2.waitKey(1)

    return hasFrames


# vid_capture = cv2.VideoCapture('/home/ubuntu/Code/openpose/videos_1/alo/20200212135117_alo.mp4')
vid = sorted(glob.glob('/home/ubuntu/Code/openpose/build/examples/tutorial_api_python/videos_fromWeb/*'+palavra+'*'))
print(vid)
for arq0 in range(len(vid)):
    vid2 = sorted(glob.glob(str(vid[arq0]))) 
    for arq2 in range(len(vid2)):
        rand = random.randint(1000000, 9999999)
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = open('output_json_icom_fromWeb/'+palavra+'_'+str(render_threshold)+'_'+stamp+'_'+str(rand)+".json","w") 
        vid_capture = cv2.VideoCapture(vid2[arq2])
        # print(vid2[arq2])
        loc =0
        posic0x = .5
        posic0y = .6
        corpo = []
        difx0=0
        dify0=0
        difx=0
        dify=0

        sec = 0
        frameRate = 0.085 #it will capture image in each 0.5 second
        success = True
        num_frame = 1
        while success and num_frame<30:
            num_frame +=1
            sec = sec + frameRate
            sec = round(sec, 2)
            try:
                success = getFrame(sec)
            except:
                pass
            print(vid2[arq2], success, num_frame)


