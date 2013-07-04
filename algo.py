import scipy.io
import numpy as np
import numpy.matlib

project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
training_directory = 'training1'
sample = 'Sample00001'
mat = scipy.io.loadmat('%s/%s/%s/%s_data.mat'%(project_directory, training_directory, sample, sample)) 

 

