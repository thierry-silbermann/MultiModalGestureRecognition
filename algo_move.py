import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re
import pickle
from algo import VideoMat

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
        

def getAllWav():
    wav_list = []
    project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav"):
                 wav_list.append(os.path.join(r,files))
    return wav_list
    
def getOneWav():
    project_dir = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    training_dir = 'validation2' #'training1'
    sample = 'Sample00532' #'Sample00001'
    return [project_dir+'/'+training_dir+'/'+sample+'/'+sample]

def plot(array, joints, labels):

    size = len(array)
    
    for data in array:
        print '###################'
        data = np.transpose(data)
        for i in range(data.shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            print joints[i] #, np.mean(data[i]), np.std(data[i])
            plt.plot(data[i], 'b.')
            ax.set_title(joints[i])
            for value in labels:
                if value != 0: 
                    name, tup = value
                    plt.axvline(x=(tup[0]-1), color='r')
                    plt.axvline(x=(tup[1]-1), color='g')
            #plt.axis([0, 1254, -1, 1])
            
            #Comment next two lines if you don't want full size screen plot
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            
            plt.show()

def plot_all_joint_from_one_sample(sk):
    print 'WorldPosition XYZ'
    plot(sk.WP, sk.joints, sample.labels)
    print 'WorldRotation XYZW' 
    plot(sk.WR, sk.joints, sample.labels)
    print 'PixelPosition 1 and 2'
    plot(sk.PP, sk.joints, sample.labels)
    
def plot_one_joint_from_batch_sample(joint_batch):
    batch = len(joint_batch)
    plt.figure(1)
    
    for index, (sk, sample) in enumerate(joint_batch):
        #get specific joint
        data = sk.WP[0] # Keep joint on WP_X
        plt.subplot(batch * 100 + 10 + index + 1)
        plt.plot(data)
        
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def main():

    
    #wav_list = pickle.load(open('vrac/All_WAV_List.pkl','rb')) 
    wav_list = getAllWav()
    #wav_list = getOneWav() # for testing purpose
    
    batch = 4 # maximum for batch size is 9
    joint_batch = [0]*batch
    
    for index, wav in enumerate(wav_list):
        path = re.sub('\_audio.wav$', '', wav)
        print path
        
        sample = VideoMat(path)
        sk = Skelet(sample)
        
        joint_batch[index%batch] = (sk, sample) 
        #plot_all_joint_from_one_sample(sk, sample)
        
        if (index+1)%batch == 0:
            plot_one_joint_from_batch_sample(joint_batch)
   
main()

