import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%pylab inline
import os
from skimage import io
#from os.path import join as pjoin
#import sys
#import copy
import pprint
import matplotlib.pyplot as plt
import PoseEstimation as PE
import detect_face
#import nn4 as network
import random
import pprint
import math
#import sklearn

#from sklearn.externals import joblib
def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def doTfMtcnnDetect():
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.8 ]  # three steps's threshold
  factor = 0.709 # scale factor

  print('Creating networks and loading parameters')
    
  with tf.Graph().as_default():       
      sess = tf.Session()#(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
      with sess.as_default():
          pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

  while True:
    frame = io.imread('cut.jpg')
    bounding_boxes, landmarks = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

    for face_position in bounding_boxes:

        face_position=face_position.astype(int)
        cv2.rectangle(frame, (face_position[0], 
                        face_position[1]), 
                  (face_position[2], face_position[3]), 
                  (0, 255, 0), 2)
    #io.imsave('save.jpg',frame)

    facenum = len(landmarks.transpose())
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,"mark facenum:",
                (0,50),
                font, 1, (255,255,255),2)

    cv2.putText(frame,str(facenum),
                (200,50),
                font, 1, (255,255,255),2)

    if(facenum > 0):

      print('face num is %d'%(facenum))
      for i in range(facenum):
        landmark_mtcnn=shapeMtcnnLandmark(landmarks.transpose()[i])
        print('landmark_mtcnn')
        pprint.pprint(landmark_mtcnn)
        points=np.asarray(landmarks.transpose()[i], dtype='float32')
        points=points.reshape((2,5))

        #img = io.imread(image_path)
        if len(landmark_mtcnn):
            Pose_Para = PE.poseEstimation5pt(frame, points)
            #print('===============%f============'%(Pose_Para[0]*180/math.pi))
            #print('pos %d %d'%(landmark_mtcnn[0][0],landmark_mtcnn[0][1]))
            print('ANGLE: %f'%(Pose_Para[0]*180/math.pi))
            cv2.putText(frame,str(Pose_Para[0]*180/math.pi),
                (int(landmark_mtcnn[0][0]), int(landmark_mtcnn[0][1])-30),
                font, 0.5, (255,255,255),2)
        else:
            print('NO face detected!')
    io.imsave('save.jpg',frame)
  #return bounding_boxes,points

def shapeMtcnnLandmark(landmarksIn):
    pprint.pprint(landmarksIn)
    xy=[]
    for i in range(5):
        xy.append((landmarksIn[i],landmarksIn[i+5]))
    return xy

if __name__ == '__main__':
    doTfMtcnnDetect()
    """
    io.imsave('save.jpg',frame)
    #pprint.pprint(frame)
    print('doTfDetect')
    boxes,landmarks=doTfMtcnnDetect(frame)
    
    print('boxex=====')
    pprint.pprint(boxes)
    print('landmarks===')
    pprint.pprint(landmarks.transpose())
    facenum = len(landmarks.transpose())

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,"mark facenum:",
                (0,50),
                font, 1, (255,255,255),2)

    cv2.putText(frame,str(facenum),
                (200,50),
                font, 1, (255,255,255),2)
    print('face num is %d'%(facenum))
    for i in range(facenum):
      landmark_mtcnn=shapeMtcnnLandmark(landmarks.transpose()[i])
      print('landmark_mtcnn')
      pprint.pprint(landmark_mtcnn)
      points=np.asarray(landmarks.transpose()[i], dtype='float32')
      points=points.reshape((2,5))

      #img = io.imread(image_path)
      if len(landmark_mtcnn):
          Pose_Para = PE.poseEstimation5pt(frame, points)
          #print('===============%f============'%(Pose_Para[0]*180/math.pi))
          #print('pos %d %d'%(landmark_mtcnn[0][0],landmark_mtcnn[0][1]))
          print('ANGLE: %f'%(Pose_Para[0]*180/math.pi))
          cv2.putText(frame,str(Pose_Para[0]*180/math.pi),
              (int(landmark_mtcnn[0][0]), int(landmark_mtcnn[0][1])-30),
              font, 0.5, (255,255,255),2)
      else:
          print('NO face detected!')
    io.imsave('save.jpg',frame)  
    """  






