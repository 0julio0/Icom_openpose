# Importing OpenCV
import cv2
import numpy as np
import argparse 
import os
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videos_fromWeb/ajuda_6edec0ea-01f0-4e4f-ba2f-29c3d977795a_5f136acd45a6890018ae40ea.mp4')
# cap = cv2.VideoCapture('cafe.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str,
# 	default=os.path.sep.join(["images", "adrian.jpg"]),
# 	help="path to input image that we'll apply GrabCut to")
ap.add_argument("-c", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=10,
    varThreshold=3,
    detectShadows=False) 

    
# Read the video
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    FRAME_WIDTH  = int(cap.get(3)/3) # float
    FRAME_HEIGHT = int(cap.get(4)/3) # float
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # Converting the image to grayscale.    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    ####################
    # # Using the Canny filter to get contours
    # edges = cv2.Canny(gray, 20, 30)
    # # Using the Canny filter with different parameters
    # edges_high_thresh = cv2.Canny(gray, 60, 120)
    # # Stacking the images to print them together
    # # For comparison
    # images = np.hstack((gray, edges, edges_high_thresh))

    # # Display the resulting frame
    # # cv2.imshow('Frame', images)
    # # cv2.imshow('Frame', edges)
    # cv2.imshow('Frame', edges_high_thresh)



    # ###################
    # #Smoothing without removing edges.
    # gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    # # cv2.imshow('Frame', gray_filtered)
    # # Applying the canny filter
    # edges = cv2.Canny(gray_filtered, 60, 120)
    # # edges = cv2.Canny(gray, 20, 30)
    # # edges_filtered = cv2.Canny(edges, 60, 120)

    # edges_filtered = cv2.Canny(gray_filtered, 60, 120)


    # # Stacking the images to print them together for comparison
    # # images = np.hstack((gray, edges, edges_filtered))
    # cv2.imshow('Frame', edges_filtered)

    





    ####################
    # edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    # foreground = fgbg.apply(edges_foreground)
    # # cv2.imshow('Frame', foreground)
    
    # # # Smooth out to get the moving area
    # # # kernel = np.ones((50,50),np.uint8)
    # kernel = np.ones((50,50),np.uint8)
    
    # foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Frame', foreground)

    # # # Applying static edge extraction
    # edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    # edges_filtered = cv2.Canny(edges_foreground, 60, 120)

    # # Crop off the edges out of the moving area
    # cropped = (foreground // 255) * edges_filtered
    # # cropped = (foreground // 255) * gray
    # cv2.imshow('Frame', cropped)





    mask = np.zeros(frame.shape[:2],np.uint8)
    rect = (101, 23, 286, 368)
    # rect = (161,79,150,150)
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # cv2.grabCut(gray, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(frame,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    frame = frame*mask2[:,:,np.newaxis]
    roughOutput = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('Frame', frame)
    # cv2.imshow('Frame', roughOutput)


    ###################
    #Smoothing without removing edges.
    gray = cv2.cvtColor(roughOutput, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    # cv2.imshow('Frame', gray_filtered)
    # Applying the canny filter
    edges = cv2.Canny(gray_filtered, 60, 120)
    # edges = cv2.Canny(gray, 20, 30)
    # edges_filtered = cv2.Canny(edges, 60, 120)

    edges_filtered = cv2.Canny(gray_filtered, 60, 120)


    # Stacking the images to print them together for comparison
    # images = np.hstack((gray, edges, edges_filtered))
    cv2.imshow('Frame', edges_filtered)




    # cv2.imshow('Frame', frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()