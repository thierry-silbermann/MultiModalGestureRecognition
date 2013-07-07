import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pickle
import re
from algo import VideoMat

# Return a list of path of every wav file present in project_directory
def getAllWav(flter):
    wav_list = []
    project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav") and flter in os.path.join(r,files):
                 wav_list.append(os.path.join(r,files))
    wav_list.sort()
    return wav_list

###########

def get_data(wav_file):
    return scipy.io.wavfile.read(wav_file) 

###########

# Home made algorithm to find interval for action in sound file
def get_interval(data):
    std = 2*np.std(data) + np.mean(data) 
    interval = []
    i = 0
    while i < len(data):
        if math.fabs(data[i]) > std:
            beg = i
            end = 0
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
            if(end - beg > 4000):
                if beg - 3000 > 0:
                    beg -= 3000
                if end + 3000 < len(data):
                    end += 3000
                interval.append((beg, end))
        i += 1
    return interval

###########

def plot(data, interval, labels, numFrames):
    t = np.zeros(data.shape[0]) + 2*np.std(data)
    coeff = data.shape[0] / numFrames
        
    # Plotting of interval find by get_interval(data)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data, 'r', t,'b', -t, 'b')
    for beg, end in interval:
        plt.axvline(x=beg, color='g')
        plt.axvline(x=end, color='y')

    # Plotting of solution interval contain in labels
    plt.subplot(212)
    plt.plot(data)
    print 'label size', len(labels), labels
    for value in labels:
        if value != 0: 
            name, tup = value
            plt.axvline(x=(tup[0]-1)*coeff, color='g')
            plt.axvline(x=(tup[1]-1)*coeff, color='y')
    plt.show()

###########

def framing(data, beg, end):
        limit = 16000 #36000
        
        if (end - beg) > limit: # crop interval
            middle = int((end-beg)/2)
            frame = data[ middle - (limit/2) : middle + (limit/2) ]
        else:
            frame = np.zeros(limit, dtype=np.int16)
            inter = end - beg
            a = (limit - inter)/2
            frame[a:a+inter] = data[beg:end]
        return frame

###########

def create_validation(data, interval):
    
    arr = [0]*len(interval)
    print len(interval)
    #print '########################'
    for i, (beg, end) in enumerate(interval):
        frame = framing(data, beg, end)
        arr[i] = frame

    return arr

###########

def create_training(data, interval, labels, numFrames):
    
    arr = [0]*len(interval)
    gestures = {'vattene':0, 'vieniqui':1, 'perfetto':2, 'furbo':3, 'cheduepalle':4,
            'chevuoi':5, 'daccordo':6, 'seipazzo':7, 'combinato':8, 'freganiente':9, 
            'ok':10, 'cosatifarei':11, 'basta':12, 'prendere':13, 'noncenepiu':14,
            'fame':15, 'tantotempo':16, 'buonissimo':17, 'messidaccordo':18, 'sonostufo':19}
    
    coeff = data.shape[0] / numFrames
    
    print '########################'

    for i, (beg, end) in enumerate(interval):
        frame = framing(data, beg, end)
        index_right_label = -1
        distance = 10000000
        for value in labels:
            if value != 0: 
                name, tup = value
                curr_beg_label = (tup[0]-1)*coeff
                if abs(beg - curr_beg_label) < distance:
                    distance = abs(beg - curr_beg_label)
                    name_right_label = name
                #print value, distance,  abs(beg - curr_beg_label), len(interval), (beg, end)
                #print name_right_label
        arr[i] = (gestures[name_right_label], frame)

    return arr

###########

def getTrainingFile(filename):

    wav_list = getAllWav('training')
    print len(wav_list)
    f = open(filename,"a")
    f.write("7077 16000 20\n")
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print path
        sample = VideoMat(path)
        rate, data = get_data(wav)
        interval = get_interval(data)
        #print rate, data.shape, len(interval)

        #print sample.labels
        a = create_training(data, interval, sample.labels, sample.numFrames)
        print len(a)
        for tupl in a:
            target, np_array = tupl
            #target = np.array(target).reshape(1,)
            #print tupl
            #print target, np_array
            #print type(target), type(np_array), np_array.dtype
            f.write(" ".join(map(str, np_array)))
            f.write("\n")
            vec_target = -np.ones(20, dtype=np.int8)
            vec_target[target] = 1
            f.write(" ".join(map(str, vec_target)))
            f.write("\n")
        #plot(data, interval, sample.labels)
    f.close()
    
    #pickle.dump(training[0:2000], open('vrac/16000_training_sound_0000_2000.pkl','wb'))
    #pickle.dump(training[2000:4000], open('vrac/16000_training_sound_2000_4000.pkl','wb'))
    #pickle.dump(training[4000:6000], open('vrac/16000_training_sound_4000_6000.pkl','wb'))
    #pickle.dump(training[6000:7077], open('vrac/16000_training_sound_6000_7077.pkl','wb'))

def getValidationFile(filename):
    wav_list = getAllWav('validation')
    print len(wav_list)
    f = open(filename,"a")
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print path[-4:],
        rate, data = get_data(wav)
        interval = get_interval(data)
        a = create_validation(data, interval)
        for np_array in a:
            f.write(" ".join(map(str, np_array)))
            f.write("\n")
    f.close()
    
def main():
    #getTrainingFile('cpp/sound_training.data')
    getValidationFile('cpp/sound_validation.data')

    
main()

