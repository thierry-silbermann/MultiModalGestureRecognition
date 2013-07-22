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
    
    
def create_prediction_file(filename):

    gestures = {'vattene':1, 'vieniqui':2, 'perfetto':3, 'furbo':4, 'cheduepalle':5,
            'chevuoi':6, 'daccordo':7, 'seipazzo':8, 'combinato':9, 'freganiente':10, 
            'ok':11, 'cosatifarei':12, 'basta':13, 'prendere':14, 'noncenepiu':15,
            'fame':16, 'tantotempo':17, 'buonissimo':18, 'messidaccordo':19, 'sonostufo':20}

    wav_list = getAllWav('training')
    print len(wav_list)
    f = open(filename,"wb")
    f.write('Id,Sequence\n')
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print path
        sample = VideoMat(path)
        labels = sample.labels
        f.write(path[-4:]+',')
        for value in labels:
            if value != 0: 
                name, tup = value
                f.write('%d '%(gestures[name]))
        f.write('\n')
    f.close()
    
create_prediction_file('training_prediction.csv')
