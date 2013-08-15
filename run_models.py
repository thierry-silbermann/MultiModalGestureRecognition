from models import movement_interval
import pandas as pd
import numpy as np


def train_audio_models(train_on, predict_on):
    pass


def train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
        path_to_audio_intervals, path_to_movement_model_with_audio_interval):

    def pad(sample):
        n_to_fill = sample.frame.min()
        if n_to_fill * 2 > len(sample):
            return sample
        new_df = sample.head(n_to_fill * 2).copy()
        new_df.ix[:, :] = np.nan
        new_df.sample_id = sample.sample_id.unique()[0]
        new_df.frame = np.hstack([np.arange(0, n_to_fill),
            np.arange(sample.shape[0], sample.shape[0] + n_to_fill)])
        sample = pd.concat([sample, new_df], axis=0)
        sample.sort('frame', inplace=True)
        sample.fillna(method='bfill', inplace=True)
        sample.fillna(method='ffill', inplace=True)
        return sample

    df_out = movement_interval(window_shift=1, retrain=True,
            train_on=train_on)
    df_out = df_out.groupby('sample_id').apply(pad)
    middle = pd.read_csv(path_to_audio_intervals, names=['sample_id', 'frame'])
    middle_probs = pd.merge(middle, df_out, how='left', on=['sample_id', 'frame'])
    middle_probs.to_csv(path_to_movement_model_with_audio_interval)


def extract_audio_intervals_from_movement_model():
    pass


def merge_models():
    pass


def create_prediction_file():
    pass


if __name__ == '__main__':

    path_to_audio_intervals = 'insert_path'
    path_to_movement_model_with_audio_interval = 'middle_proba_added.csv'
    train_on = ['training1', 'training2', 'training3', 'training4', 
            'validation1_lab', 'validation2_lab', 'validation3_lab']
    predict_on = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6']

    train_audio_models(train_on, predict_on, path_to_audio_intervals)

    train_movement_model_and_merge_on_audio_interval(train_on, predict_on,
            path_to_audio_intervals, path_to_movement_model_with_audio_interval)

