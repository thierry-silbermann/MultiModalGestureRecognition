import scipy.io
import numpy as np
import numpy.matlib

class VideoMat:

    def __init__(self, sample, isLabels):
        
        mat = scipy.io.loadmat('%s_data.mat'%(sample)) 

        self.numFrames = mat['Video']['NumFrames'][0][0][0][0]
        self.frameRate = mat['Video']['FrameRate'][0][0][0][0]
        self.frames = mat['Video']['Frames'][0][0][0]
        self.maxDepth = mat['Video']['MaxDepth'][0][0][0][0]
        
        if(isLabels): # no label for validation set, only for training
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
        

