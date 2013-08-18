from models import movement_interval
from pandas import DataFrame
import pandas as pd
import numpy as np


def train_audio_models(train_on, predict_on):
    pass


def train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
        path_to_audio_intervals, path_to_movement_model_with_audio_interval):

    def pad(sample):
        nr_frames = sample.frame.max() + 40
        out = DataFrame({'frame': np.arange(nr_frames)})
        out['sample_id'] = sample.sample_id.unique()[0]

        out = pd.merge(out, sample, how='outer', on=['sample_id', 'frame'])
        out.fillna(method='ffill', inplace=True, limit=2)
        out.fillna(method='bfill', inplace=True)
        out.fillna(method='ffill', inplace=True)

        n_to_fill = sample.frame.min()
        #if n_to_fill * 2 > len(sample):
            #return sample
        #new_df = sample.head(n_to_fill * 2).copy()
        #new_df.ix[:, :] = np.nan
        #new_df.sample_id = sample.sample_id.unique()[0]
        #new_df.frame = np.hstack([np.arange(0, n_to_fill),
            #np.arange(sample.shape[0], sample.shape[0] + n_to_fill)])
        #sample = pd.concat([sample, new_df], axis=0)
        #sample.sort('frame', inplace=True)
        ## account for 4 frame shift
        #sample.fillna(method='ffill', inplace=True, limit=2)
        #sample.fillna(method='bfill', inplace=True)
        sample.fillna(method='ffill', inplace=True)
        return out

    df_out = movement_interval(window_shift=4, retrain=True,
            train_on=train_on, predict_on=predict_on)
    middle = pd.read_csv(path_to_audio_intervals, skiprows=1, header=False)
    middle = middle.ix[:, [0, 2]]
    #middle = middle[[0, 2]]
    middle.columns = ['sample_id', 'frame']
    print middle.shape
    df_out = df_out.groupby('sample_id').apply(pad)
    #print df_out.shape
    #print df_out.head()
    print middle.shape
    print middle.head()
    middle_probs = pd.merge(middle, df_out, how='left', on=['sample_id', 'frame'])
    #print middle_probs.head()
    middle_probs = middle_probs.drop(['sample_id', 'frame', 0, 'movement'], axis=1)

    middle_probs.to_csv(path_to_movement_model_with_audio_interval, index=False)


def extract_audio_intervals_from_movement_model():
    pass


def merge_models():
    pass


def create_prediction_file():
    pass


if __name__ == '__main__':

    path_to_audio_intervals = 'Submission_table_v3.csv'
    path_to_movement_model_with_audio_interval = 'leaderboard_on_audio.csv'
    #train_on = ['training1', 'training2', 'training3', 'training4', 
    #        'validation1_lab', 'validation2_lab', 'validation3_lab']
    train_on = ['training1', 'training2', 'training3', 'training4']
    #predict_on = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6']
    predict_on = ['validation1_lab', 'validation2_lab', 'validation3_lab']

    #train_audio_models(train_on, predict_on, path_to_audio_intervals)

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)

