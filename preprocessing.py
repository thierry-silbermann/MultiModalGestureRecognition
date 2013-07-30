import numpy as np
import numpy.matlib
import pandas as pd
from pandas import DataFrame
import tarfile
import zipfile
import scipy.io
import os
from joblib import Memory, Parallel, delayed
memory = Memory('cache/')


def skeletion_from_mat(mat):

    df = DataFrame()
    Frames = mat['Video']['Frames'][0][0][0]

    has_labels = mat['Video']['Labels'][0][0].shape[1] > 0

    if has_labels:
        mat_labels = mat['Video']['Labels'][0][0][0]
        labels = np.array(['break'] * len(Frames))

        for label in mat_labels:
            name, start, end = label
            labels[start:end] = name


    for i, frame in enumerate(Frames):

        curr_frame = frame['Skeleton'][0][0]
        World_Position = DataFrame(curr_frame['WorldPosition'], columns=['x_p', 'y_p', 'z_p']) #numpy array (20,3) 
        World_Rotation = DataFrame(curr_frame['WorldRotation'], columns=['x_r', 'y_r', 'z_r', 'w_r']) #numpy array (20,4) 
        Joint_Type = DataFrame([str(name[0]) for name in curr_frame['JointType'].flatten()], columns=['JointType']) #numpy array (20, 1) 
        Pixel_Position = DataFrame(curr_frame['PixelPosition'], columns=['x_pix', 'y_pix']) #numpy array (20, 2) 
        tmp = pd.concat([World_Position, World_Rotation, Joint_Type, Pixel_Position], axis=1) 
        tmp['frame'] = i
        if has_labels:
            tmp['gesture'] = labels[i]
        df = pd.concat([df, tmp], axis=0)

    return df[df.JointType != '[]']


@memory.cache
def skeletion_from_archive(filename, is_test=False, verbose=False):

    file_path = 'data/raw_data/' + filename + '.tar.gz'
    print file_path
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
                    os.remove(fname)
                    tmp = skeletion_from_mat(mat)

                    tmp['sample_id'] = fname.split('_')[0]
                    df = pd.concat([df, tmp], axis=0)
        if is_test:
            break
        # magic trick to save memory
        tar_file.members = []
    return df


def extract_skeletion_from_files(file_names=['training1',
                                             'training2',
                                             'training3',
                                             'training4',
                                             'validation1',
                                             'validation2',
                                             'validation3'], is_test=False):

    #from preprocessing import skeletion_from_archive
    #Parallel(n_jobs=2, verbose=5)(
    #  delayed(skeletion_from_archive)(file_name, is_test=True) for file_name in file_names)

    for file_name in file_names:
        skeletion_from_archive(file_name, is_test=is_test)


if __name__ == '__main__':

    extract_skeletion_from_files()
