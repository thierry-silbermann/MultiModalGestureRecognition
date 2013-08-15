import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random
import re
import pickle
from VideoMat import VideoMat

class Skelet:

    def __init__(self, sample):
    
        nF = sample.numFrames
        fR = sample.frameRate
        
        WorldPosition_X = np.zeros((nF,fR))
        WorldPosition_Y = np.zeros((nF,fR))
        WorldPosition_Z = np.zeros((nF,fR))
        
        WorldRotation_X = np.zeros((nF,fR))
        WorldRotation_Y = np.zeros((nF,fR))
        WorldRotation_Z = np.zeros((nF,fR))
        WorldRotation_W = np.zeros((nF,fR))
        
        Position_1 = np.zeros((nF,fR))
        Position_2 = np.zeros((nF,fR))
        
        joints = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 
            'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 
            'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 
            'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']
        
        for index_frame, frame in enumerate(sample.frames):
            
            curr_frame = frame['Skeleton'][0][0]
            
            Joint_Type = curr_frame['JointType']     #numpy.array(20,1)
            dic_JT = {}
            for index_jt, joint in enumerate(Joint_Type):
                dic_JT[str(joint[0][0])] = index_jt

            World_Position = np.transpose(curr_frame['WorldPosition']) #numpy.array(20,3)
            World_Rotation = np.transpose(curr_frame['WorldRotation']) #numpy.array(20,4)
            Pixel_Position = np.transpose(curr_frame['PixelPosition']) #numpy.array(20,2)
            
            if len(dic_JT)!=1:
                for index_joint, i in enumerate(joints):
                    WorldPosition_X[index_frame][index_joint] = World_Position[0][dic_JT[i]]
                    WorldPosition_Y[index_frame][index_joint] = World_Position[1][dic_JT[i]]
                    WorldPosition_Z[index_frame][index_joint] = World_Position[2][dic_JT[i]]
                
                    WorldRotation_X[index_frame][index_joint] = World_Rotation[0][dic_JT[i]]
                    WorldRotation_Y[index_frame][index_joint] = World_Rotation[1][dic_JT[i]]
                    WorldRotation_Z[index_frame][index_joint] = World_Rotation[2][dic_JT[i]]
                    WorldRotation_W[index_frame][index_joint] = World_Rotation[3][dic_JT[i]]
                    
                    Position_1[index_frame][index_joint] = Pixel_Position[0][dic_JT[i]]
                    Position_2[index_frame][index_joint] = Pixel_Position[1][dic_JT[i]]

            if(World_Position.shape != (3, 20)):
                print 'Error WP'
            if(World_Rotation.shape != (4, 20)):
                print 'Error WR'
            if(Joint_Type.shape != (20, 1)):
                print 'Error JT'
            if(Pixel_Position.shape != (2, 20)):
                print 'Error PP'
        
        
        # WorldPosition, WorldRotation, Position with always the same format numFrames x 20
        # Each joint is always in the same place in each matrix
        self.joints = joints
        self.WP = [WorldPosition_X, WorldPosition_Y, WorldPosition_Z]
        self.WR = [WorldRotation_X, WorldRotation_Y, WorldRotation_Z, WorldRotation_W]
        self.PP = [Position_1, Position_2]
        

