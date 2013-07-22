import scipy.io.wavfile
import copy
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pickle
import random
import re
from algo import VideoMat
from Skelet import Skelet

# Return a list of path of every wav file present in project_directory
def getAllWav(flter):
    wav_list = []
    project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav") and flter in os.path.join(r,files):
                 wav_list.append(os.path.join(r,files))
    #random.shuffle(wav_list)
    wav_list.sort()
    return wav_list

###########

def get_data(wav_file):
    return scipy.io.wavfile.read(wav_file)
    
###########

def smooth_plus(x,window_len=50,window='hanning'): 
    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]
    
###########

# Home made algorithm to find interval for action in sound file
def get_interval(data):
    data = np.absolute(data)
    data = smooth_plus(data, 3000) #smooth(data, 300)
    std = 0.30*np.std(data) #+ np.mean(data) 
    interval = []
    i = 0
    while i < len(data):
        if math.fabs(data[i]) > std:
            beg, end = i, 0
            count = 0
            while (math.fabs(data[i]) > std or count < 3000) and ((i+1) <= (len(data)-1)):
                i += 1
                #print i
                if (math.fabs(data[i]) < std):
                    count += 1
                else:
                    count = 0
                    end = i
            #print beg, end, end-beg
            if(end - beg > 5000):
                if beg - 4000 > 0:
                    beg -= 4000
                if end + 5000 < len(data):
                    end += 4000
                    i += 5000
                interval.append((beg, end))
        i += 1
    return interval
    
#############

def framing(data, beg, end):
    limit = 50
    if (end - beg) > limit: # crop interval
        middle = int((end-beg)/2)
        frame = data[ middle - (limit/2) : middle + (limit/2) ]
    else:
        frame = np.zeros(limit)
        inter = end - beg
        a = (limit - inter)/2
        frame[a:a+inter] = data[beg:end]
    return frame

###########

def create_training(data, sk, interval, numFrames, labels):
    
    arr = []
    gestures = {'vattene':0, 'vieniqui':1, 'perfetto':2, 'furbo':3, 'cheduepalle':4,
            'chevuoi':5, 'daccordo':6, 'seipazzo':7, 'combinato':8, 'freganiente':9, 
            'ok':10, 'cosatifarei':11, 'basta':12, 'prendere':13, 'noncenepiu':14,
            'fame':15, 'tantotempo':16, 'buonissimo':17, 'messidaccordo':18, 'sonostufo':19}
    
    print '########################'

    data_joints = [ np.transpose(sk.WP[0])[ 6], np.transpose(sk.WP[0])[ 7], 
                    np.transpose(sk.WP[0])[10], np.transpose(sk.WP[0])[11], 
                    np.transpose(sk.WP[1])[ 6], np.transpose(sk.WP[1])[ 7], 
                    np.transpose(sk.WP[1])[10], np.transpose(sk.WP[1])[11], 
                    
                    np.transpose(sk.WR[0])[6], np.transpose(sk.WR[0])[10], 
                    np.transpose(sk.WR[1])[6], np.transpose(sk.WR[1])[10],
                    np.transpose(sk.WR[3])[7], np.transpose(sk.WR[3])[ 7],
                    
                    np.transpose(sk.PP[0])[ 6], np.transpose(sk.PP[0])[ 7], 
                    np.transpose(sk.PP[0])[10], np.transpose(sk.PP[0])[11], 
                    np.transpose(sk.PP[1])[ 6], np.transpose(sk.PP[1])[ 7], 
                    np.transpose(sk.PP[1])[10], np.transpose(sk.PP[1])[11] ]
    time_windows = 50
    frame = [0] * (time_windows*len(data_joints))
    coeff = data.shape[0] / numFrames
        
    for value in labels:
        if value != 0: 
            name, tup = value
            beg = tup[0]-1
            if(beg-5 > 0):
                beg -= 5
            end = tup[1]-1
            boolean = 0
            for beg_int, end_int in interval:
                beg_int /= coeff 
                end_int /= coeff 
            
                if beg_int > beg-5 and end_int < end+10:
                    
                    boolean = 1
                    for j, d in enumerate(data_joints):        
                        #data = np.abs(d-np.median(d)) ## data = d ##### Test with that and change in validation too
                        data = d
                        if j == 14 or j == 16 or j == 18 or j == 20:
                            data.clip(max=50)
                            data /= 50
                        if j == 15 or j == 17 or j == 19 or j == 21:
                            data.clip(max=150)
                            data /= 150
                        frame[j*time_windows:(j+1)*time_windows] = framing(data, beg, end)
            if boolean:
                arr.append((gestures[name], copy.copy(frame)))
            #else:
                #print 'Nothing'

    return arr

###########

def create_validation(data, sk, interval, numFrames):

    time_windows = 50
    arr = []
    data_joints = [ np.transpose(sk.WP[0])[ 6], np.transpose(sk.WP[0])[ 7], 
                    np.transpose(sk.WP[0])[10], np.transpose(sk.WP[0])[11], 
                    np.transpose(sk.WP[1])[ 6], np.transpose(sk.WP[1])[ 7], 
                    np.transpose(sk.WP[1])[10], np.transpose(sk.WP[1])[11], 
                    
                    np.transpose(sk.WR[0])[6], np.transpose(sk.WR[0])[10], 
                    np.transpose(sk.WR[1])[6], np.transpose(sk.WR[1])[10],
                    np.transpose(sk.WR[3])[7], np.transpose(sk.WR[3])[ 7],
                    
                    np.transpose(sk.PP[0])[ 6], np.transpose(sk.PP[0])[ 7], 
                    np.transpose(sk.PP[0])[10], np.transpose(sk.PP[0])[11], 
                    np.transpose(sk.PP[1])[ 6], np.transpose(sk.PP[1])[ 7], 
                    np.transpose(sk.PP[1])[10], np.transpose(sk.PP[1])[11] ]
    
    frame = [0] * (time_windows*len(data_joints))
    
    #print '###', len(np.transpose(sk.WP[0])[5])-time_windows, (len(np.transpose(sk.WP[0])[5])-time_windows)/5

    coeff = data.shape[0] / numFrames
    for beg, end in interval:
        beg /= coeff #apply coeff
        end /= coeff #apply coeff
        for j, d in enumerate(data_joints):        
            #data = np.abs(d-np.median(d)) ## data = d ##### Test with that and change in training too
            data = d
            if j == 14 or j == 16 or j == 18 or j == 20:
                data.clip(max=50)
                data /= 50
            if j == 15 or j == 17 or j == 19 or j == 21:
                data.clip(max=150)
                data /= 150
            frame[j*time_windows:(j+1)*time_windows] = framing(data, beg, end)
        arr.append(copy.copy(frame))

    return arr
    
###########

def getTrainingFile(filename):

    wav_list = getAllWav('training')
    print len(wav_list)
    f = open(filename,"wb")
    #7107 number of interval
    f.write("7107 1100 20\n") #Change 1100 (50('time_frame') * len(data_joints))
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print path
        sample = VideoMat(path)
        sk = Skelet(sample)
        rate, data = get_data(wav)
        interval = get_interval(data)
        a = create_training(data, sk, interval, sample.numFrames, sample.labels)
        print len(a)
        for tupl in a:
            target, np_array = tupl
            f.write(" ".join(map(str, np_array)))
            f.write("\n")
            vec_target = -np.ones(20, dtype=np.int8)
            vec_target[target] = 1
            f.write(" ".join(map(str, vec_target)))
            f.write("\n")
        #plot(data, interval, sample.labels)
    f.close()
    

def getValidationFile(filename):
    create_submission_file = open('create_submission2.txt', 'wb')
    wav_list = getAllWav('training') # validation
    print len(wav_list)
    f = open(filename,"wb")
    #wav_list = ['/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition/training1/Sample00001/Sample00001_audio.wav']
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print path[-4:],
        sample = VideoMat(path)
        sk = Skelet(sample)
        rate, data = get_data(wav)
        interval = get_interval(data)
        a = create_validation(data, sk, interval, sample.numFrames)
        print len(a)
        create_submission_file.write('%s %d\n' %(path[-4:], len(a)))
        for np_array in a:
            f.write(" ".join(map(str, np_array)))
            f.write("\n")
        #break
    f.close()
    create_submission_file.close()
    
def main():
    #getTrainingFile('cpp/movement_training.data')
    getValidationFile('cpp/movement_validation2.data')

    
main()

