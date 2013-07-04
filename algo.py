import scipy.io
import numpy as np
import numpy.matlib

project_directory = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
training_directory = 'training1'
sample = 'Sample00001'
mat = scipy.io.loadmat('%s/%s/%s/%s_data.mat'%(project_directory, training_directory, sample, sample)) 

for key in mat.keys():
    print "key: %s" % (key)
    #print "value: %s" %(mydictionary[key]) #not a good idea to uncomment !

"""
key: __version__
key: Video
key: __header__
key: __globals__
"""

Video_info = mat['Video']

print mat['Video'].dtype
#[('NumFrames', 'O'), ('FrameRate', 'O'), ('Frames', 'O'), ('MaxDepth', 'O'), ('Labels', 'O')]

NumFrames = mat['Video']['NumFrames'][0][0][0][0]
FrameRate = mat['Video']['FrameRate'][0][0][0][0]

Frames = mat['Video']['Frames'][0][0][0]

print Frames.shape
# (1254,)

for frame in Frames:
    curr_frame = frame['Skeleton'][0][0]

    World_Position = curr_frame['WorldPosition'] #numpy array (20,3)
    World_Rotation = curr_frame['WorldRotation'] #numpy array (20,4)
    Joint_Type = curr_frame['JointType'] #numpy array (20, 1)
    Pixel_Position = curr_frame['PixelPosition'] #numpy array (20, 2)
    
print 'End'
