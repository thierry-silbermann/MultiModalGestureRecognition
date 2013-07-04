import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re
from algo import VideoMat

def getAllWav():
    wav_list = []
    project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav"):
                 wav_list.append(os.path.join(r,files))
    return wav_list

def get_data(wav_file):
    return scipy.io.wavfile.read(wav_file) 

def get_interval(data):
    std = 2*np.std(data) + np.mean(data) 
    interval = []
    i = 0
    while i < len(data):
        if math.fabs(data[i]) > std:
            beg = i
            end = 0
            count = 0
            while math.fabs(data[i]) > std or count < 3000 and i+1 < len(data):
                i += 1
                if (math.fabs(data[i]) < std):
                    count += 1
                else:
                    count = 0
                    end = i
            #print beg, end, end-beg
            if(end - beg > 4000):
                interval.append((beg, end))
        i += 1
    return interval

def plot(data, interval, labels):
    t = np.zeros(data.shape[0]) + 2*np.std(data)
    print data.shape[0]
    print labels[19][1][1]
    coeff = data.shape[0] / labels[19][1][1]
        
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data, 'r', t,'b', -t, 'b')
    for beg, end in interval:
        plt.axvline(x=beg, color='g')
        plt.axvline(x=end, color='y')

    plt.subplot(212)
    plt.plot(data)
    for name, tup in labels:
        plt.axvline(x=(tup[0]-1)*coeff, color='g')
        plt.axvline(x=(tup[1]-1)*coeff, color='y')
    plt.show()


def main():
    project_dir = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    training_dir = 'training1'
    sample = 'Sample00001'

    wav_list = getAllWav()
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        sample = VideoMat(path)
        rate, data = get_data(wav)
        interval = get_interval(data)
        print rate, data.shape, len(interval)
        print sample.labels
        plot(data, interval, sample.labels)

main()

