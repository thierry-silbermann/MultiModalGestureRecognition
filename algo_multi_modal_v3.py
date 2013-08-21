#!/usr/bin/env python

__author__ = "Thierry Silbermann"
__credits__ = ["Thierry Silbermann", "Immanuel Bayer"]
__email__ = "thierry.silbermann@gmail.com"

import copy
import math
import os
import pickle
import random
import re
import time

from VideoMat import VideoMat
from Skelet import Skelet
#from Head_interaction import Head_inter
import mfcc as mf

import numpy as np
import scipy.io.wavfile
from scipy.signal import argrelextrema
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from numpy import genfromtxt

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
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]

###########
# Return a list of path of every wav file present in project_directory

def getAllWav(flter, isSorted, root_directory):
    wav_list = []
    project_directory = root_directory
    for r,d,f in os.walk(project_directory):
        for files in f:
            if files.endswith(".wav") and '/'+flter+'/' in os.path.join(r,files):
                 wav_list.append(os.path.join(r,files))
    if isSorted:
        wav_list.sort()
    else:
        random.shuffle(wav_list)
    return wav_list

###########

def getOneWav(root_directory, training_dir, sample):
    return [root_directory+'/'+training_dir+'/'+sample+'/'+sample+'_audio.wav']

###########

def get_data(wav_file):
    return scipy.io.wavfile.read(wav_file) 

###########

def true_peak(arr, data):
    if len(arr) < 2:
        return arr, [data[x] for x in arr]
    else:
        new_arr = []
        tmp = []
        curr_arr = arr[0]
        for index in range(1, len(arr)):
            #print np.mean(data[arr[index-1]:arr[index]]), min(data[arr[index-1]], data[arr[index]])
            if np.mean(data[arr[index-1]:arr[index]] + 0.1 < min(data[arr[index-1]], data[arr[index]])):
                if arr[index-1] not in new_arr:
                    new_arr.append(arr[index-1])
                new_arr.append(arr[index])
            else:
                if arr[index-1] not in tmp:
                    new_arr.append(arr[index-1])
                
        return new_arr, [data[x] for x in new_arr]

############

# Homemade algorithm to find interval for action in sound file
def get_interval(data, numFrames):
    data = np.absolute(data)
    data = smooth_plus(data, 3000) #smooth(data, 300)
    std = 0.30*np.std(data) #+ np.mean(data) 
    interval = []
    coeff = data.shape[0] / numFrames
    i = 0
    while i < len(data):
        if math.fabs(data[i]) > std:
            beg, end = i, 0
            count = 0
            while (math.fabs(data[i]) > std or count < 3000) and ((i+1) <= (len(data)-1)):
                i += 1
                if (math.fabs(data[i]) < std):
                    count += 1
                else:
                    count = 0
                    end = i
            if(end - beg > 3000):
                if end + 10000 < len(data):
                    end += 10000
                    i += 6500
                else:
                    end = len(data) - 1
                interval.append(['', (1+(beg/coeff), 1+(end/coeff))])
        i += 1
    return interval

###########


def interval_analysis(interval, sk):
    HandL_X, HandL_Y = get_specific_data(sk.PP, 'HandLeft', sk.joints)
    HandR_X, HandR_Y = get_specific_data(sk.PP, 'HandRight', sk.joints)
    
    #Y coordinate contains the interesting informations
    HL_max = np.median(HandL_Y)
    HR_max = np.median(HandR_Y)
    
    for i in xrange(len(interval) - 1, -1, -1):
        value = interval[i]
        name, (beg, end) = value
        delete = True
        if (HandL_Y[beg-1:end-1]==0).sum() > 0: #be sure that we have data from sensors
            delete = False
        else:
            if HL_max > max(HandL_Y[beg-1:end-1]):
                if (HL_max - min(HandL_Y[beg-1:end-1])) > 30:
                    delete = False
            else:
                if (max(HandL_Y[beg-1:end-1]) - min(HandL_Y[beg-1:end-1])) > 30:
                    delete = False
            if HR_max > max(HandR_Y[beg-1:end-1]):
                if (HR_max - min(HandR_Y[beg-1:end-1])) > 30:
                    delete = False
            else:
                if (max(HandR_Y[beg-1:end-1]) - min(HandR_Y[beg-1:end-1])) > 30:
                    delete = False
        #delete
        if delete:
            print 'delete interval:', (beg, end)
            del interval[i]
    #print 'Before merge:', interval
    new_interval = []
    i = 1
    while i < len(interval):
        name, (beg1, end1) = interval[i-1]
        name, (beg2, end2) = interval[i]
        if end1 + 2 >= beg2: #overlapping sequence
            
            if (HL_max - min(HandL_Y[end1-5:end1])) > 30 and (HL_max - min(HandL_Y[beg2:beg2+5])) > 30:
                    new_interval.append((name, (beg1, end2)))
                    print 'Merge interval on left:', (beg1, end1), (beg2, end2)
                    i += 1
            
            elif (HR_max - min(HandR_Y[end1-5:end1])) > 30 and (HR_max - min(HandR_Y[beg2:beg2+5])) > 30:
                new_interval.append((name, (beg1, end2)))
                print 'Merge interval on right:', (beg1, end1), (beg2, end2)
                i += 1
            else:
                new_interval.append(interval[i-1])
        else:
            new_interval.append(interval[i-1])
        i += 1
    if i == len(interval):
        new_interval.append(interval[i-1])
    return new_interval

###########

def get_specific_data(Coordinate, joint, joints):
    data1 = np.transpose(Coordinate[0])
    data2 = np.transpose(Coordinate[1])
    if (data1.shape != data2.shape):
        print 'Error'
    for i in range(data1.shape[0]):
        if joints[i] == joint:
            return data1[i], data2[i]

##########

def euclidian_dist(AX, AY, BX, BY, normalization=1):
    return (np.square(np.square(AX-BX) + np.square(AY-BY)))/(normalization+0.00001)
   
##########     

def create_features(data, labels, numFrames, sk):
    
    #print labels
    t = np.zeros(data.shape[0]) + 2*np.std(data)
    coeff = data.shape[0] / numFrames

    a_name = ['WorldPositionX', 'WorldPositionY', 'WorldRotationX', 'WorldRotationY', 'PixelPosition1', 'PixelPosition2']
    a = [sk.WP[0],sk.WP[1],sk.WR[0],sk.WR[1],sk.PP[0],sk.PP[1]]
    
    ############
    # Construct feature to detect distance between head and hand/elbow
    Head_data_X, Head_data_Y            = get_specific_data(sk.PP, 'Head', sk.joints)
    HandLeft_data_X, HandLeft_data_Y    = get_specific_data(sk.PP, 'HandLeft', sk.joints)
    HandRight_data_X, HandRight_data_Y  = get_specific_data(sk.PP, 'HandRight', sk.joints)
    HipCenter_data_X, HipCenter_data_Y  = get_specific_data(sk.PP, 'HipCenter', sk.joints)
    ElbowLeft_data_X, ElbowLeft_data_Y  = get_specific_data(sk.PP, 'ElbowLeft', sk.joints)
    ElbowRight_data_X, ElbowRight_data_Y = get_specific_data(sk.PP, 'ElbowRight', sk.joints)
    
    norm_Head_Hip = euclidian_dist(Head_data_X, Head_data_Y, HipCenter_data_X, HipCenter_data_Y)
    
    dist_Head_HandLX = euclidian_dist(Head_data_X, 0, HandLeft_data_X, 0, norm_Head_Hip)
    dist_Head_HandLY = euclidian_dist(Head_data_Y, 0, HandLeft_data_Y, 0, norm_Head_Hip)
    dist_Head_HandRX = euclidian_dist(Head_data_X, 0, HandRight_data_X, 0, norm_Head_Hip)
    dist_Head_HandRY = euclidian_dist(Head_data_X, 0, HandRight_data_Y, 0, norm_Head_Hip)
    dist_Head_ElbowLX = euclidian_dist(Head_data_X, 0, ElbowLeft_data_X, 0, norm_Head_Hip)
    dist_Head_ElbowLY = euclidian_dist(Head_data_Y, 0, ElbowLeft_data_Y, 0, norm_Head_Hip)
    dist_Head_ElbowRX = euclidian_dist(Head_data_X, 0, ElbowRight_data_X, 0, norm_Head_Hip)
    dist_Head_ElbowRY = euclidian_dist(Head_data_Y, 0, ElbowRight_data_Y, 0, norm_Head_Hip)
    
    features = [dist_Head_HandLX, dist_Head_HandLY, 
                dist_Head_HandRX, dist_Head_HandRY, 
                dist_Head_ElbowLX, dist_Head_ElbowLY, 
                dist_Head_ElbowRX, dist_Head_ElbowRY]
    ###############
    
    
    nb_feat_per_joint = 19
    nb_joint = 6
    number_of_column = 1 + nb_feat_per_joint*nb_joint + len(features)
    number_of_line = len([value for value in labels if value != 0])
    data_frame = [[0]*number_of_column for x in xrange(number_of_line)]
    #print number_of_line, number_of_column
    
    joints = sk.joints
    #data_mixed = np.zeros(np.transpose(a[0])[0].shape[0])
    for index, array in enumerate(a):

        size = len(array)
        color = {'ElbowLeft':'r', 'ElbowRight':'k',
                 'HipCenter':'c',
                 'WristLeft':'g','HandLeft':'b',
                 'WristRight':'m','HandRight':'y'}

        data = np.transpose(array)
        ShouldCenter_data = np.zeros(data[0].shape)
        for i in range(data.shape[0]):
            if joints[i] == 'HipCenter':
                c = copy.copy(data[i])
                c -= np.median(c)
                c_min = [-0.4, -0.4, -3, -1, -150, -250]
                c_max = [0.5, 0.8, 0.5, 1.5, 130, 150]
                c = (c - c_min[index]) / (c_max[index] - c_min[index])
                HipCenter_data = c
            
        
        data_mixed = np.zeros(data[0].shape[0])
        j = -1
        for i in range(data.shape[0]):
            if a_name[index] in ['WorldPositionX', 'WorldPositionY', 'PixelPosition1', 'PixelPosition2']:
                kept_joint = ['ElbowLeft', 'ElbowRight', 'WristLeft', 'HandLeft', 'WristRight', 'HandRight'] #Should got ride of Wrist joint, it duplicates Hand info
            elif a_name[index] in ['WorldRotationX', 'WorldRotationY']:
                kept_joint = ['ElbowLeft', 'ElbowRight', 'WristLeft', 'HandLeft', 'WristRight', 'HandRight']
            if joints[i] in kept_joint:
                j += 1
                data[i][data[i]==0]=np.median(data[i])
                 
                b = copy.copy(data[i])
                b -= np.median(b)

                b_min = [-0.4, -0.4, -3, -1, -150, -250]
                b_max = [0.5, 0.8, 0.5, 1.5, 130, 150]
                b = (b - b_min[index]) / (b_max[index] - b_min[index])
                
                #b = 2*b - 1    #re-scale between -1 and 1
                b = b - HipCenter_data
                if a_name[index] in ['WorldRotationX', 'WorldRotationY']:
                    b = -(b)
                index_data_frame = 0
                for value in labels:
                    if value != 0: 
                        name, tup = value

                        if a_name[index] in ['WorldPositionX']:
                            data_frame[index_data_frame][0] = name
                            
                            res = ((b[tup[0]-1:(tup[1]-1)] < -0.1)).sum() 
                            res_min = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_min = str(min(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 1] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 2] = res_min
                            
                            res = ((b[tup[0]-1:(tup[1]-1)] > 0.1)).sum() 
                            res_max = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_max = str(max(b[tup[0]-1:(tup[1]-1)]))
                            
                            data_frame[index_data_frame][j * nb_feat_per_joint + 3] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 4] = res_max
                            
                            index_data_frame += 1
                            
                        if a_name[index] in ['WorldPositionY']:
                            res = ((b[tup[0]-1:(tup[1]-1)] < -0.1)).sum() 
                            res_min = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_min = str(min(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 5] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 6] = str(min(b[tup[0]-1:(tup[1]-1)]))
                            
                            res = ((b[tup[0]-1:(tup[1]-1)] > 0.1)).sum() 
                            res_max = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_max = str(max(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 7] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 8] = str(max(b[tup[0]-1:(tup[1]-1)]))
                            
                            index_data_frame += 1
                        
                        if a_name[index] in ['WorldRotationX']:
                            res = ((b[tup[0]-1:(tup[1]-1)] > 0.1)).sum() 

                            peak = argrelextrema(b[(tup[0]-1):(tup[1]-1)], np.greater)
                            new_peak = []
                            for nb in peak[0]: 
                                if b[tup[0]-1+nb] > 0.1:
                                    new_peak.append(nb)
                            real_peak = []
                            max_value = 0
                            if(len(new_peak)>0):
                                real_peak, peak_value = true_peak([fd+(tup[0]-1) for fd in new_peak], b)
                                max_value = max(peak_value)
                                #print color[joints[i]], value, [fd+(tup[0]-1) for fd in new_peak], (real_peak, max_value)
                            
                            data_frame[index_data_frame][j * nb_feat_per_joint + 9] = str(len(real_peak))
                            data_frame[index_data_frame][j * nb_feat_per_joint + 10] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 11] = str(max_value)
                            index_data_frame += 1
                            
                        if a_name[index] in ['WorldRotationY']:
                            res = ((b[tup[0]-1:(tup[1]-1)] < -0.1)).sum() 
                            res_min = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_min = str(min(b[tup[0]-1:(tup[1]-1)]))
                            data_frame[index_data_frame][j * nb_feat_per_joint + 12] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 13] = res_min
                            index_data_frame += 1
                            
                        if a_name[index] in ['PixelPosition1']:
                            res = ((b[tup[0]-1:(tup[1]-1)] < -0.1)).sum() 
                            res_min = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_min = str(min(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 14] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 15] = res_min
                            
                            res = ((b[tup[0]-1:(tup[1]-1)] > 0.1)).sum() 
                            res_max = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_max = str(max(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 16] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 17] = res_max
                            
                            index_data_frame += 1    
                        
                        if a_name[index] in ['PixelPosition2']:
                            res = ((b[tup[0]-1:(tup[1]-1)] < -0.1)).sum() 
                            res_min = '0'
                            if res>0:
                                #print color[joints[i]], value, res
                                res_min = str(min(b[tup[0]-1:(tup[1]-1)]))
                                
                            data_frame[index_data_frame][j * nb_feat_per_joint + 18] = str(res)
                            data_frame[index_data_frame][j * nb_feat_per_joint + 19] = res_min
                            
                            index_data_frame += 1     
                data_mixed += np.abs(b-np.median(b)) #np.abs((data[i]-np.median(data[i])))
    
    
    # Construct feature to detect distance between head and hand
                
    for index, value in enumerate(labels):
        if value != 0: 
            name, tup = value
            for index_feat, feat in enumerate(features):
                tmp_sort = copy.copy(feat[tup[0]-1:(tup[1]-1)])
                tmp_sort.sort()
                if (tup[1] - tup[0]) > 10:
                    tmp_sort = tmp_sort[:10]
                data_frame[index][nb_feat_per_joint*nb_joint + index_feat + 1] = str(np.mean(tmp_sort))
            
                if (np.mean(tmp_sort) == np.nan):
                    print index_feat, tmp_sort

    return data_frame
    

###########    

def train_model_on_gestures(wav_list):

    gestures = {'vattene':0, 'vieniqui':1, 'perfetto':2, 'furbo':3, 'cheduepalle':4,
                    'chevuoi':5, 'daccordo':6, 'seipazzo':7, 'combinato':8, 'freganiente':9, 
                    'ok':10, 'cosatifarei':11, 'basta':12, 'prendere':13, 'noncenepiu':14,
                    'fame':15, 'tantotempo':16, 'buonissimo':17, 'messidaccordo':18, 'sonostufo':19}

    dataX = []
    i = 0
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print '\n', '##############'
        print path[-25:]
        sample = VideoMat(path, True)
        sk = Skelet(sample)
        rate, data = get_data(wav)
        data_frame = np.asarray(create_features(data, sample.labels, sample.numFrames, sk))
        #print 'data_frame !', data_frame.shape
        #data_frame2 = np.asarray(Head_inter(path, sample.labels).data_frame)
        #data_frame = np.hstack((data_frame, data_frame2))
        dataX += copy.copy(data_frame)
        
        
    # 1 target / 19 * 6 joints infos / 8 Head/Hand distances / 5 Head box = 128 features
    #Train model: Don't use the Head box features, don't really improve the model  
    data_frame = np.asarray(dataX)
    Y = data_frame[:, 0]
    Y = np.asarray([gestures[i] for i in Y])
    X = data_frame[:, 1:]
    X = X.astype(np.float32, copy=False)
    X = X[:, :122] 
    clf = RandomForestClassifier(n_estimators=300, criterion='entropy', min_samples_split=10, 
            min_samples_leaf=1, verbose=2, random_state=1) #n_jobs=2
    clf = clf.fit(X, Y)
    pickle.dump(clf, open('gradient_boosting_model_gestures.pkl','wb'))
 
#################### 
 
def train_model_on_sound(wav_list): 
    gestures = {'vattene':0, 'vieniqui':1, 'perfetto':2, 'furbo':3, 'cheduepalle':4,
                'chevuoi':5, 'daccordo':6, 'seipazzo':7, 'combinato':8, 'freganiente':9, 
                'ok':10, 'cosatifarei':11, 'basta':12, 'prendere':13, 'noncenepiu':14,
                'fame':15, 'tantotempo':16, 'buonissimo':17, 'messidaccordo':18, 'sonostufo':19}
    dataX = []
    
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        print '\n', '##############'
        print path[-25:]
        sample = VideoMat(path, True)
        sk = Skelet(sample)
        rate, data = get_data(wav)
        labels = sample.labels
        coeff = data.shape[0] / sample.numFrames
        
        interval = get_interval(data, sample.numFrames) #comment to use true interval data
        interval = interval_analysis(interval, sk)
        interval = [['', (beg, end)] for name, (beg, end) in interval if end-beg>10]
        
        features_nb = 2588 #Change to add mspec and spec features
        for value in labels:
            if value != 0:
                name, (beg, end) = value
                for inter in interval:
                    name2, (beg2, end2) = inter
                    if beg2>beg-5 and end2<end+5:
                        space = end2 - beg2 - 9
                        limit = 40
                        data_interval = np.zeros(40*coeff)
                        if(space > limit):
                            data_interval = data[(beg2-1)*coeff:(beg2+limit-1)*coeff]
                        else:
                            data_interval[:space*coeff] = data[(beg2-1)*coeff:(end2-10)*coeff]
                        ceps, mspec, spec = mf.mfcc(data_interval)
                        #print ceps.shape, mspec.shape, spec.shape
                        data_tmp = np.zeros(features_nb)
                        data_tmp[0] = gestures[name]
                        data_tmp[1:2588] = ceps.reshape(2587)
                        #data_tmp[2588:10548] = mspec.reshape(7960)
                        #data_tmp[1273:13561] = spec.reshape(12288)
                        data_tmp = np.nan_to_num(data_tmp)
                        dataX.append(copy.copy(data_tmp))
                        break
    print len(dataX)
    data = np.asarray(dataX)
    Y = data[:, 0]
    X = data[:, 1:2588]
    X = X.clip(min=-100)
    clf = GradientBoostingClassifier(n_estimators=200, verbose=2, max_depth=7, min_samples_leaf=10, min_samples_split=20, random_state=0)
    clf = clf.fit(X, Y) 
    pickle.dump(clf, open('gradient_boosting_model_sound.pkl','wb'))
    
    clf = RandomForestClassifier(n_estimators=300, criterion='entropy', min_samples_split=10, min_samples_leaf=1, verbose=2, random_state=1) #n_jobs=2
    clf = clf.fit(X, Y) 
    pickle.dump(clf, open('random_forest_model_sound.pkl','wb'))
    
    clf = ExtraTreesClassifier(n_estimators=300, min_samples_split=10, min_samples_leaf=1, verbose=2, random_state=1) #n_jobs=2
    clf = clf.fit(X, Y) 
    pickle.dump(clf, open('extra_trees_model_sound.pkl','wb'))
       
###########                
            
def create_predicting_feature(path, wav, clf_gb, clf_rf, gradient_boosting_model_gestures):

    print '\n', '##############'
    print path[-25:]
    sample = VideoMat(path, False)
    sk = Skelet(sample)
    rate, data = get_data(wav)
    coeff = data.shape[0] / sample.numFrames
    labels = get_interval(data, sample.numFrames)
    labels = interval_analysis(labels, sk) 
    labels = [['', (beg, end)] for name, (beg, end) in labels if end-beg>10]
    data_frame = np.asarray(create_features(data, labels, sample.numFrames, sk))
    #data_frame2 = np.asarray(Head_inter(path, labels).data_frame)
    #data_frame = np.hstack((data_frame, [str(end-beg) for name, (beg, end) in labels]))
    #data_frame = np.hstack((data_frame, data_frame2))
    X_test = np.asarray(data_frame)
    X_test = X_test[:, 1:123]   # shape(x, 122)
    X_test = X_test.astype(np.float32, copy=False)
    class_proba = gradient_boosting_model_gestures.predict_proba(X_test)
    print 'nb of labels', len(labels)
    
    def get_class_proba_sound(clf_gb, clf_rf, data, interval, numFrames):
    
        coeff = data.shape[0] / numFrames
        features_nb = 2587 
        X = np.zeros((len(interval), features_nb))
        for i, inter in enumerate(interval):
            name, (beg, end) = inter
            space = end - beg - 9
            #print 'space:', space
            limit = 40
            data_interval = np.zeros(40*coeff)
            if(space > limit):
                data_interval = data[(beg-1)*coeff:(beg+limit-1)*coeff]
            else:
                data_interval[:space*coeff] = data[(beg-1)*coeff:(end-10)*coeff]
            ceps, mspec, spec = mf.mfcc(data_interval)
            #print 'ceps, mspec, spec', ceps.shape, mspec.shape, spec.shape
            X[i, :2587] = ceps.reshape(2587)
            #X[i, 2587:10547] = mspec.reshape(7960)
            #X[i, 10547:22835] = spec.reshape(12288)
        X = np.nan_to_num(X)
        X = X.clip(min=-100)
        
        return clf_gb.predict_proba(X), clf_rf.predict_proba(X)
    
    class_proba_gb, class_proba_rf = get_class_proba_sound(clf_gb, clf_rf, data, labels, sample.numFrames)
    
    return class_proba, class_proba_gb, class_proba_rf, labels

def blend_model(wav_list, submission_table_filename):

    clf_gb_sound = pickle.load(open('gradient_boosting_model_sound.pkl','rb'))
    clf_rf_sound = pickle.load(open('random_forest_model_sound.pkl','rb'))
    clf_gb_gest = pickle.load(open('gradient_boosting_model_gestures.pkl','rb'))

    output = open(submission_table_filename,'wb', ) #Submission.csv
    output.write('Id,Sequence\n') 
    
    for wav in wav_list:
        path = re.sub('\_audio.wav$', '', wav)
        class_proba, class_proba_gb, class_proba_rf, labels = create_predicting_feature(path, wav, clf_gb_sound, clf_rf_sound, clf_gb_gest)
        for i in range(class_proba.shape[0]):
            name, (beg, end) = labels[i] 
            output.write('%s,%s,%d,%s,%s,%s\n' %(path[-11:], path[-4:], (end+beg)/2, 
                                                      ','.join(  (map(str, class_proba[i]))  ), 
                                                      ','.join(  (map(str, class_proba_gb[i]))  ),
                                                      ','.join(  (map(str, class_proba_rf[i]))  ) ) ) 
        print class_proba.shape, class_proba_gb.shape, class_proba_rf.shape
        #print class_proba, class_proba_gb, class_proba_rf
        if(class_proba.shape != class_proba_gb.shape or class_proba.shape != class_proba_rf.shape):
            raise Exception("Error dimension between class proba")

    output.close()

def submission(submission_table_filename):
    data = genfromtxt(submission_table_filename, delimiter=',', skip_header=1)

    #print data.shape
    #print np.isnan(data).sum()
    data = np.nan_to_num(data)
    
    ID = data[:, 1]
    Frame = data[:, 2]
    
    uniq_ID = np.unique(ID)

    Model1 = data[:, 3:23]
    Model2 = data[:, 23:43]
    Model3 = data[:, 43:63]
    
    if data.shape[1] == 83:
        print 'Four models blending'
        Model4 = data[:, 63:83]
        w = [0, 0.1, 0.4, 0.5] 
        threshold = 0.25
        final_proba = w[0] * Model1 + w[1] * Model2 + w[2] * Model3 + w[3] * Model4
    else:
        print 'Three models blending'
        w = [0.4, 0.3, 0.3] #[1./3, 1./3, 1./3] 
        threshold = 0.4
        final_proba = w[0] * Model1 + w[1] * Model2 + w[2] * Model3 
        print 'Threshold:', threshold
        print 'Weight:', w
    
    output = open('Kaggle_Submission.csv','wb', ) #Submission.csv
    output.write('Id,Sequence\n') 
    for i in uniq_ID:
        output.write('%s,' %(str(int(i)).zfill(4)))
        index = np.where(ID==i)[0]
        class_proba = final_proba[index]
    
        ## Write prediction that are sure (greater than the threshold of 0.4)
        actual_gesture = -1
        nb_of_detection = class_proba.shape[0]
        for i, gestures_proba in enumerate(class_proba):
            
            max_value = np.amax(gestures_proba)
            index_max_value = np.argmax(gestures_proba)+1
            #print gestures_proba, max_value, index_max_value
            
            if (max_value >= threshold):   # Only keep high probability match
                if actual_gesture != index_max_value:    #No repetition
                    actual_gesture = index_max_value
                    if i==nb_of_detection-1:
                        output.write('%d'%(index_max_value))
                    else:
                        output.write('%d '%(index_max_value))
        output.write('\n')
    output.close()

def main():
    root = 'data/raw_data' #/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition/

    #Training part
    wav_list = []
    for directory in ['training1', 'training2', 'training3', 'training4', 'validation1_lab', 'validation2_lab', 'validation3_lab']: #'validation1_lab', 'validation2_lab', 'validation3_lab'
        wav_list += getAllWav(directory, True, root)
    wav_list.sort() #Just in case

    print '=> Features creation and training on gestures: 20mn'
    train_model_on_gestures(wav_list)
    
    print '=> Features creation and training on sound: 4h18'
    train_model_on_sound(wav_list)

    #Predicting part
    wav_list = []
    for directory in ['test1', 'test2', 'test3', 'test4', 'test5', 'test6']: #, 'validation2', 'validation3' #test1, test2, test3, test4, test5, test6
        wav_list += getAllWav(directory, True, root)
    wav_list.sort() #Just in case
    
    #wav_list = getOneWav(root, 'test3', 'Sample00917') # Uncomment this line and 
                                                            # comment the previous three line to do prediction on one sample
    
    print '=> Full prediction: 41mn'
    submission_table_filename = 'Submission_table.csv'
    blend_model(wav_list, submission_table_filename)

    #submission_table_filename = 'final_Submission_table.csv'
    submission(submission_table_filename)
    
    print 'See Submission.csv for prediction'
    
if __name__ == "__main__":
    main()

