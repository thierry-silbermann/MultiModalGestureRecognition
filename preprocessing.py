import numpy as np
import numpy.matlib
import pandas as pd
from pandas import DataFrame
import tarfile
import zipfile
import scipy.io
import postprocessing
import pandas as pd
import os
from joblib import Memory, Parallel, delayed
memory = Memory('cache/')


def skeletion_from_mat(mat, is_test_file):

    df = DataFrame()
    Frames = mat['Video']['Frames'][0][0][0]

    if is_test_file:
        has_labels = False
    else:
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


#@memory.cache
def skeletion_from_archive(filename, is_test=False, verbose=False):

    file_path = 'data/raw_data/' + filename + '.tar.gz'
    is_test_file = 'test' in filename
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
                    tmp = skeletion_from_mat(mat,is_test_file)

                    tmp['sample_id'] = fname.split('_')[0]
                    df = pd.concat([df, tmp], axis=0)
        if is_test:
            break
        # magic trick to save memory
        tar_file.members = []
    return df
skeletion_from_archive_cached = memory.cache(skeletion_from_archive)


def extract_skeletion_from_files(file_names=['training1',
                                             'training2',
                                             'training3',
                                             'training4',
                                             'validation1_lab',
                                             'validation2_lab',
                                             'validation3_lab'], is_test=False):

    from preprocessing import skeletion_from_archive_cached
    Parallel(n_jobs=-1, verbose=5)(
      delayed(skeletion_from_archive_cached)(file_name) for file_name in file_names)

   # for file_name in file_names:
   #     skeletion_from_archive(file_name, is_test=is_test)


@memory.cache
def sequence_truth(file_names=['training1',
                               'training2',
                               'training3',
                               'training4']):
    df_out = DataFrame()
    for file_name in file_names:
        df = skeletion_from_archive_cached(file_name)
        df = df[['frame', 'gesture', 'sample_id']].dropna().drop_duplicates()
        df_out = pd.concat((df_out, df))
    return df_out


@memory.cache
def preprocessed_skeleton(file_name, demain=True, keep_only_top_40=True,
        train_id=True, drop_lower_joints=True, dummy_gesture=False,
        window_shift=1, window_length=40):

    df = skeletion_from_archive_cached(file_name)

    if demain:
        def demean(arr):
            num_cols =  ['w_r', 'x_p', 'x_pix', 'x_r', 'y_p', 'y_pix',
                    'y_r', 'z_p', 'z_r']
            arr[num_cols] = arr[num_cols] - arr[num_cols].mean()
            return arr #- arr.mean()

        df = df.groupby('sample_id').apply(demean)

    if train_id:
        def add_train_id(df):
            df = df.sort('frame')
            gestures = iter(df.gesture)
            last = gestures.next()
            count = 0
            gest_i = []
            gest_i.append(count)
            for gest in gestures:
                if gest == last:
                    gest_i.append(str(count))
                    last = gest

                else:
                    count += 1
                    gest_i.append(str(count))
                    last = gest
            df['gesture_nr'] = np.array(gest_i)
            return df
        df = df.groupby('sample_id').apply(add_train_id)

    if keep_only_top_40:
        def get_top(arr, n=window_length, column='frame'):
            start_frame = arr.frame.min()
            return arr[ (arr.frame < start_frame + n) | (arr.gesture == 'break')]

        # make sure each gesture appears only once in each sequence!!!
        df = df.groupby(df.gesture + df.sample_id + df.gesture_nr,
                group_keys=False).apply(get_top)

    if drop_lower_joints:
        joints_to_drop = ['KneeLeft', 'AnkleLeft',
           'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']

        for joint in joints_to_drop:
            df = df[df.JointType != joint]

    if dummy_gesture:
        def add_dummy_gestures(df, window_length=window_length,
                window_shift=window_shift):
            start_ = df.frame.min()
            end_ = df.frame.max()
            w_start = start_
            w_end = w_start + window_length
            windows = DataFrame()

            while w_end < end_:
                df_w = df[(df.frame >= w_start) & (df.frame <= w_end)]
                df_w['dummy_gesture'] = w_start
                windows = pd.concat([windows, df_w], axis=0)

                w_start += window_shift
                w_end += window_shift

            return windows
        df= df.groupby('sample_id', group_keys=False).apply(add_dummy_gestures)

    return df


@memory.cache
def aggregated_skeletion_win(file_names=['validation1'],
        agg_functions=['median', 'var'], window_shift=1, window_length=40):
    X = DataFrame()

    for file_name in file_names:
        df = preprocessed_skeleton(file_name, keep_only_top_40=False,
             train_id=False, dummy_gesture=True, window_shift=window_shift,
             window_length=window_length)
        df = df.drop('frame')
        df = df.groupby(['sample_id', 'dummy_gesture', 'JointType']
                ).agg(agg_functions).unstack('JointType')
        X = pd.concat([X, df])
        del df
    return X


@memory.cache
def aggregated_skeletion(file_names=['training1', 'training2', 'training3',
                    'training4'], agg_functions=['mean'], window_length=40):
    X = DataFrame()

    for file_name in file_names:
        df = preprocessed_skeleton(file_name, window_length=window_length)
        df = df.drop('frame')
        df = df.groupby(['sample_id', 'gesture', 'JointType', 'gesture_nr']
                ).agg(agg_functions).unstack('JointType')
        X = pd.concat([X, df])
        del df
    y = np.array([gesture for (_, gesture, _) in X.index])
    return X, y


if __name__ == '__main__':
    skeletion_from_archive('test1', is_test=False, verbose=True)
    #extract_skeletion_from_files(file_names=['test1', 'test2', 'test3',
    #    'test4', 'test5', 'test6'], is_test=False)
    #from preprocessing import agg_movement_intervals
    #extract_skeletion_from_files()
    #agg_movement_intervals('training1')
