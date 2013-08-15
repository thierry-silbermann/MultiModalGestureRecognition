import cv
import cv2.cv
import numpy as np
import os
import random
import re
import sys
import matplotlib.pyplot as plt

from VideoMat import VideoMat
from Skelet import Skelet

class Head_inter:

    def get_specific_data(self, Coordinate, joint, joints):
        self.data1 = np.transpose(Coordinate[0])
        self.data2 = np.transpose(Coordinate[1])
        if (self.data1.shape != self.data2.shape):
            print 'Error'
        for i in range(self.data1.shape[0]):
            if joints[i] == joint:
                return self.data1[i], self.data2[i]

    def DetectFace(self, image, x, y, w, h, boolean=True):
        if boolean:
            if x-40 > 0:
                x -= 40
            if y-30 > 0:
                y -= 30
        cv.Rectangle(image, (x,y),(x+w,y+h), cv.RGB(155, 255, 25), 2)
        cv.ResetImageROI(image)
        return image, (x, y, w, h)

    def __init__(self, wav, labels):
        
        path = re.sub('\_audio.wav$', '', wav) 
        print path
        
        sample = VideoMat(path, False)
        sk = Skelet(sample)
        #labels = sample.labels
        
        Head_X, Head_Y = self.get_specific_data(sk.PP, 'Head', sk.joints)
        ShouldC_X, ShouldC_Y = self.get_specific_data(sk.PP, 'ShoulderCenter', sk.joints)
        HandL_X, HandL_Y = self.get_specific_data(sk.PP, 'HandLeft', sk.joints)
        HandR_X, HandR_Y = self.get_specific_data(sk.PP, 'HandRight', sk.joints)

        path = path+'_color.mp4' #user or color
        capture = cv.CaptureFromFile(path)

        nFrames = int(  cv.GetCaptureProperty( capture, cv.CV_CAP_PROP_FRAME_COUNT ) )
        width = int( cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH) )
        height = int( cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT) )

        w, h = 160, 160
        #An image, which will contain grayscale version of our frame
        frame_gray = cv.CreateImage((w, h), 8, 1)
        mean_array_upper = np.zeros(nFrames)
        mean_array_lower = np.zeros(nFrames)
        mean_array_neck  = np.zeros(nFrames)
        mean_array_left  = np.zeros(nFrames)
        mean_array_right = np.zeros(nFrames)

        for f in xrange(nFrames):
            img = cv.QueryFrame(capture)

            cv.Threshold(img, img, 255, 255, cv.CV_THRESH_BINARY)

            image, face = self.DetectFace(img, int(HandL_X[f]), int(HandL_Y[f]), 15, 15, False)
            image, face = self.DetectFace(img, int(HandR_X[f]), int(HandR_Y[f]), 15, 15, False)
            image, face = self.DetectFace(img, int(Head_X[f]), int(Head_Y[f]), 80, 80)
            
            pos = (face[0], face[1])
            size = (face[2], face[3])
            
            if pos[0] < 0:
                pos = (0, 0)
            if size[0] > image.width:
                size = (image.width, image.height)
                
            cropped = cv2.cv.CreateImage(size, 8, 3)
            cv2.cv.Copy(cv2.cv.GetSubRect(image, pos + size), cropped)
            image2 = cv2.cv.CreateImage( (w,h), 8, 3 )           
            cv2.cv.Resize(cropped, image2)

            cv2.cv.CvtColor(image2, frame_gray, cv.CV_RGB2GRAY)
            
            mean_array_upper[f] = np.mean( np.asarray( frame_gray[   : 70 ,   :  ] ))
            mean_array_lower[f] = np.mean( np.asarray( frame_gray[ 70:140 ,   :  ] ))
            mean_array_neck[f]  = np.mean( np.asarray( frame_gray[140:160 ,   :  ] ))
            mean_array_left[f]  = np.mean( np.asarray( frame_gray[   :    ,   :80] ))
            mean_array_right[f] = np.mean( np.asarray( frame_gray[   :    , 80:  ] ))
            
        
        #print sample.labels
        array_to_analyze = [mean_array_upper, mean_array_lower, mean_array_neck, mean_array_left, mean_array_right]
        min_value = np.apply_along_axis(np.min, 1, array_to_analyze) 
        
        number_of_column = len(array_to_analyze)
        number_of_line = len([value for value in labels if value != 0])
        data_frame = [[0]*number_of_column for x in xrange(number_of_line)]
        
        for index_interest, interest_array in enumerate(array_to_analyze):
            for index_label, value in enumerate(labels):
                if value != 0: 
                    name, tup = value
                    a = ((interest_array[tup[0]-1:(tup[1]-1)] > min_value[index_interest])).sum() 
                    data_frame[index_label][index_interest] = str(a)
        self.data_frame = data_frame

