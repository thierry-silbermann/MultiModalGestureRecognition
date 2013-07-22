import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pickle
import re
from algo import VideoMat
import copy
import mfcc as mf

def smooth(data, window=50):  
    new_data = copy.copy(data)
 
    for i in range(window, len(data)-window):  
        new_data[i] = np.sum(data[i-window:i+window])/(window-20)
    return new_data  

###########

def smooth_plus(x,window_len=50,window='hanning'): 
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

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

def get_better_interval(data, labels, numFrames):
    coeff = data.shape[0] / numFrames
    plt.subplot(311)
    plt.plot(data)
    data = np.absolute(data)
    plt.subplot(312)
    plt.plot(data)
    data = smooth(data, 300)
    plt.subplot(313)
    plt.plot(data)
    for value in labels:
        if value != 0: 
            name, tup = value
            plt.axvline(x=(tup[0]-1)*coeff, color='g')
            plt.axvline(x=(tup[1]-1)*coeff, color='y')
    plt.show()
    
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

###########

def plot(data, interval, labels, numFrames):
    t = np.zeros(data.shape[0]) + 2*np.std(data)
    coeff = data.shape[0] / numFrames
        
    # Plotting of interval find by get_interval(data)
    plt.figure(1)
    plt.subplot(311)
    plt.plot(data, 'r', t,'b', -t, 'b')
    for beg, end in interval:
        plt.axvline(x=beg, color='g')
        plt.axvline(x=end, color='y')

    # Plotting of solution interval contain in labels
    data_trans = np.copy(data)
    data_trans = np.absolute(data_trans)
    data_trans = smooth(data_trans, 300)
    std = 0.80*np.std(data_trans)
    plt.subplot(312)
    plt.plot(data_trans, 'r', t,'b')
    for beg, end in interval:
        plt.axvline(x=beg, color='g')
        plt.axvline(x=end, color='y')

    # Plotting of solution interval contain in labels
    plt.subplot(313)
    plt.plot(data)
    #print 'label size', len(labels), labels
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
    selection = 'training'
    wav_list = getAllWav(selection)
    print len(wav_list)
    #f = open(filename,"a")
    #f.write("7077 16000 20\n")
    diff = 0
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        #print path
        sample = VideoMat(path)
        rate, data = get_data(wav)
        interval = get_interval(data)
        if selection == 'training':
            true_interval = [value for value in sample.labels if value != 0]
            print len(interval), len(true_interval), path[-4:] 
            diff += math.fabs(len(interval) - len(true_interval))
        elif selection == 'validation':
            print len(interval), path[-4:] 
        plot(data, interval, sample.labels, sample.numFrames)

        '''
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
        '''
    #f.close()
    print diff, len(wav_list)


###########

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

###########    

def solution1():
    #wav_list = getAllWav('training')
    wav_list = ["/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition/training1/Sample00078/Sample00078_audio.wav"]
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        sample = VideoMat(path)
        rate, data = get_data(wav)
        #interval = get_interval(data)
        interval = get_better_interval(data, sample.labels, sample.numFrames)
        plot(data, interval, sample.labels, sample.numFrames)
    
###########
    
def main():
    getTrainingFile('cpp/sound_training.data')
    #getValidationFile('cpp/sound_validation.data')
    #solution1()
    
main()

