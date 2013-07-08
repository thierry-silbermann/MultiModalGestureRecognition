 import scipy.io
import numpy as np
import numpy.matlib
import pandas as pd
from pandas import DataFrame
import tarfile
import zipfile
import scipy.io
#import os


def skeletion_from_mat(mat):
    df = DataFrame()
    Frames = mat['Video']['Frames'][0][0][0]

    for i, frame in enumerate(Frames):
        curr_frame = frame['Skeleton'][0][0]

        World_Position = DataFrame(curr_frame['WorldPosition'], columns=['x', 'y', 'z']) #numpy array (20,3)
        World_Rotation = DataFrame(curr_frame['WorldRotation'], columns=['a', 'b', 'c', 'd']) #numpy array (20,4)
        Joint_Type = DataFrame(curr_frame['JointType'], columns=['JointType'])         #numpy array (20, 1)
        Pixel_Position = DataFrame(curr_frame['PixelPosition'], columns=['pix_x', 'pix_y']) #numpy array (20, 2)

        tmp = pd.concat([World_Position, World_Rotation, Joint_Type, Pixel_Position], axis=1)
        tmp['frame'] = i
        df = pd.concat([df, tmp], axis=0)

    return df



def skeletion_from_archive(file_path, is_test=False, verbose=False):

    df = DataFrame()
    tar_file = tarfile.open(file_path, 'r:gz')
    for tar_info in tar_file:

        if verbose:
            print tar_info

        file_ = tar_file.extractfile(tar_info)

        if zipfile.is_zipfile(file_):
            zfile = zipfile.ZipFile(file_, 'r')

            for info in zfile.infolist():

                if '.mat' in info.filename:
                    fname = info.filename
                    if verbose:
                        print fname

                    zfile.extract(fname)
                    mat=scipy.io.loadmat(fname)
                    #os.remove('/data/raw_data/' + fname)
                    tmp = skeletion_from_mat(mat)

                    tmp['sample_id'] = fname.split('_')[0]
                    df = pd.concat([df, tmp], axis=0)
        if is_test:
            break
        # magic trick to save memory
        tar_file.members = []
    return df


if __name__ == '__main__':
    file_path = 'data/raw_data/training2.tar.gz'
    skeletion_from_archive(file_path, verbose=True).head()
