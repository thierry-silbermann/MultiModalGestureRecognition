import scipy.io
import numpy as np
import numpy.matlib

class VideoMat:

    def __init__(self, sample):
        
        mat = scipy.io.loadmat('%s_data.mat'%(sample)) 

        self.numFrames = mat['Video']['NumFrames'][0][0][0][0]
        self.frameRate = mat['Video']['FrameRate'][0][0][0][0]
        self.frames = mat['Video']['Frames'][0][0][0]
        self.maxDepth = mat['Video']['MaxDepth'][0][0][0][0]
        
        if('training' in sample): # no label for validation set, only for training
            Labels = mat['Video']['Labels'][0][0][0]
            labels = [0] * 20
            for i in range(Labels.shape[0]):
                name = str(Labels[i]['Name'][0])
                begin = Labels[i]['Begin'][0][0]
                end = Labels[i]['End'][0][0]
                labels[i] = [name, (begin, end)]
            self.labels = labels
        else:
            self.labels = None
        

def main():
    project_dir = '/home/thierrysilbermann/Documents/Kaggle/11_Multi_Modal_Gesture_Recognition'
    training_dir = 'validation1'
    sample = 'Sample00410'

    a = VideoMat(project_dir+'/'+training_dir+'/'+sample+'/'+sample)
    
    for frame in a.frames:
        curr_frame = frame['Skeleton'][0][0]

        #WorldPosition: The world coordinates position structure represents the global position of a tracked joint.
         #The X value represents the x-component of the subject global position (in millimeters).
         #The Y value represents the y-component of the subject global position (in millimeters).
         #The Z value represents the z-component of the subject global position (in millimeters).
        World_Position = curr_frame['WorldPosition'] #numpy array (20,3)
        
        #WorldRotation: The world rotation structure contains the orientations of 
        #skeletal bones in terms of absolute transformations. The world rotation 
        #structure provides the orientation of a bone in the 3D camera space. 
        #Is formed by 20x4 matrix, where each row contains the W, X, Y, Z values of 
        #the quaternion related to the rotation. 
        World_Rotation = curr_frame['WorldRotation'] #numpy array (20,4)
        
        Joint_Type = curr_frame['JointType'] #numpy array (20, 1)
        Pixel_Position = curr_frame['PixelPosition'] #numpy array (20, 2)
    
main()

