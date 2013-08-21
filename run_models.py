from models import movement_interval
from pandas import DataFrame
import pandas as pd
import numpy as np
import os
from algo_multi_modal_v3 import *
from postprocessing import pad_smooth

def train_audio_models(train_on, predict_on, submission_table_filename, root):
    

    #Training part
    wav_list = []
    for directory in train_on: 
        wav_list += getAllWav(directory, True, root)
    wav_list.sort() #Just in case

    print '=> Features creation and training on gestures: 20mn'
    train_model_on_gestures(wav_list)
    
    print '=> Features creation and training on sound: 4h18'
    train_model_on_sound(wav_list)
    
    #Predicting part
    wav_list = []
    for directory in predict_on: 
        wav_list += getAllWav(directory, True, root)
    wav_list.sort() #Just in case
    
    #wav_list = getOneWav(root, 'training1', 'Sample00001') # Uncomment this line and 
                                                            # comment the previous three line to do prediction on one sample
    
    print '=> Full prediction: 41mn'
    blend_model(wav_list, submission_table_filename)

def train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
        path_to_audio_intervals, path_to_movement_model_with_audio_interval):


    df_out = movement_interval(train_on=train_on, predict_on=predict_on)
    df_out = df_out.groupby('sample_id').apply(pad_smooth, window_len=20)

    middle = pd.read_csv(path_to_audio_intervals, header=None, skiprows=1)
    middle = middle.ix[:, [0, 2]]
    middle.columns = ['sample_id', 'frame']
    middle_probs = pd.merge(middle, df_out, how='left', on=['sample_id', 'frame'])
    middle_probs = middle_probs.drop(['sample_id', 'frame', 0, 'movement'], axis=1)

    middle_probs.to_csv(path_to_movement_model_with_audio_interval, index=False)

def merge_models(path_to_audio_intervals, path_to_movement_model_with_audio_interval):
    os.system("paste -d ',' " + path_to_audio_intervals + " " + path_to_movement_model_with_audio_interval + " > final_" + path_to_audio_intervals)


def run_val_all_model():
    # parameter estimation settings
    predict_on = ['validation1_lab', 'validation2_lab', 'validation3_lab']
    train_on = ['training1', 'training2', 'training3', 'training4']
    path_to_audio_intervals = 'Submission_table_t1234_v123.csv'
    #train_audio_models(train_on, predict_on, path_to_audio_intervals, root)
    path_to_movement_model_with_audio_interval =\
            'movement_probs_added_' + path_to_audio_intervals

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)


def run_val1_model():
    # parameter estimation settings
    predict_on = ['training1', 'validation1_lab']
    train_on = ['training2', 'training3', 'training4', 'validation2_lab', 'validation3_lab']
    path_to_audio_intervals = 'Submission_table_t234v23_t1v1.csv'
    #train_audio_models(train_on, predict_on, path_to_audio_intervals, root)
    path_to_movement_model_with_audio_interval =\
            'movement_probs_added_' + path_to_audio_intervals

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)


def run_vali2_model():
    # parameter estimation settings
    predict_on = ['training2', 'validation2_lab']
    train_on = ['training1', 'training3', 'training4', 'validation1_lab', 'validation3_lab']
    path_to_audio_intervals = 'Submission_table_t134v13_t2v2.csv'
    #train_audio_models(train_on, predict_on, path_to_audio_intervals, root)
    path_to_movement_model_with_audio_interval =\
            'movement_probs_added_' + path_to_audio_intervals

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)


def run_final_model():
    root = 'data/raw_data/'
    path_to_audio_intervals = 'Submission_table_t1234v123_test123456.csv'
    path_to_movement_model_with_audio_interval = 'movement_probs_added_' + path_to_audio_intervals

    train_on = ['training1', 'training2', 'training3', 'training4', 
            'validation1_lab', 'validation2_lab', 'validation3_lab']
    predict_on = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6']

    train_audio_models(train_on, predict_on, path_to_audio_intervals, root)

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)

    merge_models(path_to_audio_intervals, path_to_movement_model_with_audio_interval)

    submission('final_'+path_to_audio_intervals) # -> Will produce a 'Kaggle_Submission.csv' file


if __name__ == '__main__':

    run_final_model()
    #run_val1_model()
    #run_vali2_model()
    #run_val_all_model()
