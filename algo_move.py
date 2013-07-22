import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random
import re
import pickle
from algo import VideoMat
from Skelet import Skelet



import pylab

def smoothListTriangle(list,strippedXs=False,degree=5):  
    weight=[]  
    window=degree*2-1  
    smoothed=[0.0]*(len(list)-window)  
    for x in range(1,2*degree):weight.append(degree-abs(degree-x))  
    w=np.array(weight)  
    for i in range(len(smoothed)):  
        smoothed[i]=sum(np.array(list[i:i+window])*w)/float(sum(w))  
    return smoothed  

def smoothListGaussian(list,strippedXs=False,degree=5):  

    window=degree*2-1  
    weight=np.array([1.0]*window)  
    weightGauss=[]  
     
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
         
    weight=np.array(weightGauss)*weight  
    smoothed=[0.0]*(len(list)-window)  
     
    for i in range(len(smoothed)):  
        smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  
    return smoothed  



# Return a list of path of every wav file present in project_directory
def getAllWav(flter):
    wav_list = []
    project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav") and flter in os.path.join(r,files):
                 wav_list.append(os.path.join(r,files))
    random.shuffle(wav_list)
    return wav_list
    
def getOneWav():
    project_dir = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    training_dir = 'validation2' #'training1'
    sample = 'Sample00532' #'Sample00001'
    return [project_dir+'/'+training_dir+'/'+sample+'/'+sample]

def plot(array, joints, labels):

    size = len(array)
    color = {'WristLeft':'g','HandLeft':'b',
             'WristRight':'m','HandRight':'y'}
    
    #['ElbowLeft', 'WristLeft', 'HandLeft', 'ElbowRight', 'WristRight', 'HandRight']
    fig = plt.figure()
    for index, data in enumerate(array):
        print '###################'
        data = np.transpose(data)
        for i in range(data.shape[0]):
            
            if joints[i] in ['WristLeft', 'HandLeft', 'WristRight', 'HandRight']:
                
                
                plt.subplot(size * 2 * 100 + 10 + 1 + 2*index)
                ax = fig.add_subplot(size * 2 * 100 + 10 + 1 + 2*index)
                #plt.plot(data[i], 'b.', ls='-')
                a = np.abs(data[i]-np.median(data[i]))
                b = smoothListGaussian(a)
                plt.plot(b, color[joints[i]], ls='-')
                ax.set_title(joints[i])
                for value in labels:
                    if value != 0: 
                        name, tup = value
                        #plt.axvline(x=(tup[0]-1), color='r')
                        #plt.axvline(x=(tup[1]-1), color='g')
                        plt.axvline(x=float(tup[1]-1+tup[0]-1)/2, color='r')
                
                
                plt.subplot(size * 2 * 100 + 10 + 2 + 2*index)
                plt.plot(data[i], color[joints[i]], ls='-')
                #print mlpy.findpeaks_dist(data[i], mindist=30)
                #print [float(tup[1]-1+tup[0]-1)/2 for (name, tup) in labels]
                for value in labels:
                    if value != 0: 
                        name, tup = value
                        #plt.axvline(x=(tup[0]-1), color='r')
                        #plt.axvline(x=(tup[1]-1), color='g')
                        plt.axvline(x=float(tup[1]-1+tup[0]-1)/2, color='r')
                
                #Comment next two lines if you don't want full size screen plot
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                
    plt.show()

def plot_all_joint_from_one_sample(sk, sample):
    print 'WorldPosition XYZ'
    #plot(sk.WP, sk.joints, sample.labels)
    print 'WorldRotation XYZW' 
    #plot(sk.WR, sk.joints, sample.labels)
    print 'PixelPosition 1 and 2'
    plot(sk.PP, sk.joints, sample.labels)
    
def plot_one_joint_from_batch_sample(joint_batch):
    batch = len(joint_batch)
    plt.figure(1)
    
    for index, (sk, sample) in enumerate(joint_batch):
        #get specific joint
        data = sk.WP[1] ###### Keep joint on WP_X
        data = np.transpose(data)
        
        joints = ['HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 
            'ElbowLeft', 'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 
            'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 
            'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']
        
        i = 6 ###### joint[i]
        plt.subplot(batch * 100 + 10 + index + 1)
        if sample.labels != None:
            for value in sample.labels:
                if value != 0: 
                    name, tup = value
                    #plt.axvline(x=float(tup[1]-1+tup[0]-1)/2, color='r') #middle
                    plt.axvline(x=float(tup[1]-1), color='r') #beginning
                    
        #plt.plot(data[i]-np.median(data[i]))
        plt.plot(np.abs(data[i]-np.median(data[i])))
        
        #plt.ylim([-0.5,0.5]) # For WP and WR
        #plt.ylim([0,50]) # For PP
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def main():

    
    #wav_list = pickle.load(open('vrac/All_WAV_List.pkl','rb')) 
    wav_list = getAllWav('training')
    #wav_list = getOneWav() # for testing purpose
    
    batch = 4 # maximum for batch size is 9
    joint_batch = [0]*batch
    
    for index, wav in enumerate(wav_list):
        path = re.sub('\_audio.wav$', '', wav)
        print path
        
        sample = VideoMat(path)
        sk = Skelet(sample)
        
        plot_all_joint_from_one_sample(sk, sample)
        
        #joint_batch[index%batch] = (sk, sample) 
        #if (index+1)%batch == 0:
        #    plot_one_joint_from_batch_sample(joint_batch)
        #if index+1%40 == 0:
        #    break
   
main()

