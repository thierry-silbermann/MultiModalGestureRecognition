from postprocessing import postprocess, dump_predictions
from preprocessing import skeletion_from_archive_cached, preprocessed_skeleton
from preprocessing import aggregated_skeletion, aggregated_skeletion_win
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from pandas import DataFrame
import numpy as np
import pandas as pd
from pandas import DataFrame
from joblib import Memory
memory = Memory('cache/')


gesture_to_id = {'vieni':2, 'break':0, 'prend':14, 'sonos':20, 'chevu':6,
        'dacco':7, 'perfe':3, 'vatte':1, 'basta':13, 'buoni':18, 'chedu':5,
        'cosat':12, 'fame':16, 'nonce':15, 'furbo':4, 'combi':9, 'frega':10,
        'seipa':8, 'tanto':17, 'messi':19, 'ok':11}


@memory.cache
def leaderboard_model(out_file='leaderboard.csv',window_shift=1,
        window_length=40, retrain=False,
        train_on=['training1','training2', 'training3', 'training4'],
        predict_on=['validation1_lab', 'validation2_lab', 'validation3_lab']):

    filename = 'cache/joblib/rf_leaderboard' + str(window_length) + '.joblib.pkl'
    #file_names=['training1','training2', 'training3', 'training4']

    if retrain:
        X, y = aggregated_skeletion(file_names=train_on,
                agg_functions=['median', 'var', 'min', 'max'],
                window_length= window_length)
        X = X.fillna(0)
        y = np.array([gesture_to_id[gest] for gest in y])


        clf = ExtraTreesClassifier(n_estimators=500, random_state=0,
            n_jobs=-1)
        clf.fit(X, y)
        _ = joblib.dump(clf, filename, compress=9)
    else:
        clf = joblib.load(filename)

    X_win = aggregated_skeletion_win(predict_on,
            agg_functions=['median', 'var', 'min', 'max'], 
        window_shift=window_shift, window_length=window_length)

    X_win = X_win.fillna(0)
    y_pred = clf.predict_proba(X_win)
    df_pred = DataFrame(y_pred, index=[s for (s, _) in X_win.index])

    to_dump = df_pred.groupby(level=0).apply(postprocess)
    dump_predictions(to_dump, out_path=out_file)
    return df_pred, to_dump


@memory.cache
def movement_interval(train_on=['training1','training2', 'training3', 'training4'],
        predict_on=['validation1_lab', 'validation2_lab', 'validation3_lab']):

    window_shift = 5
    window_length = 40

    print 'aggregated_skeletion_win'
    X_win = aggregated_skeletion_win(predict_on,
        agg_functions=['median', 'var', 'min', 'max'], 
        window_shift=window_shift, window_length=window_length)
    X_win= X_win.fillna(0)

    print 'train rf model'
    X, y = aggregated_skeletion(file_names=train_on,
            agg_functions=['median', 'var', 'min', 'max'])
    X = X.fillna(0)
    y = np.array([gesture_to_id[gest] for gest in y])

    clf = ExtraTreesClassifier(n_estimators=1500, random_state=0,
        n_jobs=-1)
    clf.fit(X, y)
    del X
    del y

    print 'rf predict'
    y_pred = clf.predict_proba(X_win)

    df_out = pd.concat([DataFrame.from_records(X_win.index.values.tolist(),
        columns=['sample_id', 'frame']), DataFrame(y_pred)], axis=1)
    df_out['movement'] = np.array(np.argmax(y_pred, axis=1) != 0,
                                                                dtype=int)
    # adjust for sliding window size
    df_out.frame = df_out.frame + 20
    return df_out


@memory.cache
def eval_seq_model(out_file='eval_model.csv',window_shift=1, retrain=False):

    filename = 'cache/joblib/rf_eval_model.joblib.pkl'
    file_names=['training1', 'training3', 'training4', 
                    'validation1_lab', 'validation3_lab']

    if retrain:
        X, y = aggregated_skeletion(file_names=file_names,
                agg_functions=['median', 'var', 'min', 'max'])
        X = X.fillna(0)
        y = np.array([gesture_to_id[gest] for gest in y])


        clf = ExtraTreesClassifier(n_estimators=500, random_state=0,
            n_jobs=-1)
        clf.fit(X, y)
        _ = joblib.dump(clf, filename, compress=9)
    else:
        clf = joblib.load(filename)

    X_win = aggregated_skeletion_win(['validation2_lab', 'training2'],
            agg_functions=['median', 'var', 'min', 'max'],
            window_shift=window_shift)

    y_pred = clf.predict_proba(X_win)
    df_pred = DataFrame(y_pred, index=[s for (s, _) in X_win.index])

    to_dump = df_pred.groupby(level=0).apply(postprocess)
    dump_predictions(to_dump, out_path=out_file)
    return df_pred, to_dump


@memory.cache
def eval_gesture_model(retrain=False, window_shift=1, window_length=40,
        train_on=['training1', 'training3', 'training4',
                    'validation1_lab', 'validation3_lab'],
        predict_on=['validation2_lab', 'training2']):

    filename = 'cache/joblib/rf_eval_model' + str(window_length) + '.joblib.pkl'
    #file_names=['training1', 'training3', 'training4',
    #                'validation1_lab', 'validation3_lab']

    if retrain:
        X, y = aggregated_skeletion(file_names=train_on,
                agg_functions=['median', 'var', 'min', 'max'],
                window_length=window_length)
        X = X.fillna(0)
        y = np.array([gesture_to_id[gest] for gest in y])


        clf = ExtraTreesClassifier(n_estimators=500, random_state=0,
            n_jobs=-1)
        clf.fit(X, y)
        _ = joblib.dump(clf, filename, compress=9)
    else:
        clf = joblib.load(filename)

    X_test, y_test = aggregated_skeletion(predict_on,
            agg_functions=['median', 'var', 'min', 'max'],
        window_length=window_length)
    X_test = X_test.fillna(0)
    y_test = np.array([gesture_to_id[gest] for gest in y_test])
    y_pred = clf.predict_proba(X_test)
    return y_pred, y_test


@memory.cache
def agg_movement_intervals(file_name, has_labels=False,
         train_on=['training1','training2', 'training3', 'training4']):
    from models import movement_interval


    df = skeletion_from_archive_cached(file_name)
    df = preprocessed_skeleton(file_name, demain=True, keep_only_top_40=False,
            train_id=False, drop_lower_joints=True, dummy_gesture=False)


    df_interval = movement_interval(train_on=train_on)
    df_merge = pd.merge(df, df_interval[['sample_id', 'frame', 'movement']], on=['frame', 'sample_id'], how='left')

    df_merge.sort(['sample_id', 'frame'], inplace=True)
    df_merge.fillna(method='ffill', inplace=True)
    df_merge.fillna(method='bfill', inplace=True)

    df_merge['interval'] = (df_merge.movement.shift(1) != df_merge.movement).astype(int).cumsum()
    # delete break frames tagged with gesture
    df_merge = df_merge[(df_merge.movement != 0)]

    # unify gestures in interval to most frequent one
    def smooth_gesture(df):
        df['gesture'] = df.gesture.value_counts().index[0]
        return df

    agg_functions=['median', 'var', 'min', 'max']

    if has_labels:
        df_merge = df_merge.groupby(['interval']).apply(smooth_gesture)
        # drop intervalls without movement or if labelt as break
        df_merge = df_merge[(df_merge['gesture'] != 'break') & (df_merge['movement'] == 1)]
        y_gest = df_merge[['sample_id', 'gesture', 'interval']].drop_duplicates()
        y_gest['gesture'] = np.array([gesture_to_id[g] for g in y_gest.gesture])

    df_merge = df_merge.drop(['movement', 'frame'], axis=1)
    df_agg = df_merge.groupby(['sample_id', 'JointType', 'interval']).agg(agg_functions).unstack('JointType')
    df_agg['_count'] = df_merge.groupby(['sample_id', 'JointType', 'interval']).agg({'x_p': 'count'}).unstack('JointType')[('x_p', 'HandRight')]

    if has_labels:
        return df_agg, y_gest
    return df_agg, None


@memory.cache
def collect_movement_intervalls(file_names=['training1', 'training2', 'training3',
                                    'training4'], has_labels=False,
         train_on=['training1','training2', 'training3', 'training4']):

    X = DataFrame()
    y = DataFrame()

    for file_name in file_names:
        df_X, df_y = agg_movement_intervals(file_name, has_labels=has_labels,
                                                        train_on=train_on)
        X = pd.concat([X, df_X])
        if has_labels:
            y = pd.concat([y, df_y])
        del df_X
        del df_y
    return X, y


if __name__ == '__main__':
    from models import leaderboard_model, eval_seq_model, eval_gesture_model

    #leaderboard_model(out_file='leaderboard60_5.csv',window_shift=5,
        #window_length=60, retrain=True)
    #leaderboard_model(out_file='leaderboard50_5.csv',window_shift=5,
        #window_length=50, retrain=True)
    #leaderboard_model(out_file='leaderboard40_5.csv',window_shift=5,
    #    window_length=40, retrain=False)
    #leaderboard_model(out_file='leaderboard30_5.csv',window_shift=5,
        #window_length=30, retrain=True)
    #leaderboard_model(out_file='leaderboard20_5.csv',window_shift=5,
        #window_length=20, retrain=True)

    #eval_gesture_model(window_shift=5,
      #  window_length=60, retrain=False)
    #eval_gesture_model(window_shift=5,
        #window_length=50, retrain=False)
    #eval_gesture_model(window_shift=5,
        #window_length=40, retrain=False)
    #eval_gesture_model(window_shift=5,
        #window_length=30, retrain=False)
    #eval_gesture_model(window_shift=5,
        #window_length=20, retrain=False)

    #leaderboard_model(retrain=True)
    #leaderboard_model(window_shift=1)
    #leaderboard_model(window_shift=5)
    #eval_seq_model(retrain=True)
    #eval_seq_model(retrain=False, window_shift=1)
    #eval_seq_model(retrain=False, window_shift=5)
    #eval_gesture_model(retrain=True)
    #eval_gesture_model(window_shift=1)
    #eval_gesture_model(window_shift=5)
    from models import movement_interval, agg_movement_intervals, collect_movement_intervalls
    movement_interval(train_on=['training1','training2','training3', 'training4'],
        predict_on=['validation1_lab', 'validation2_lab', 'validation3_lab'])

    movement_interval(train_on=['training1','training3', 'training4', 'validation1_lab',
            'validation3_lab'],
        predict_on=['training2', 'validation2_lab'])

    movement_interval(train_on=['training2','training3', 'training4', 'validation2_lab',
            'validation3_lab'],
        predict_on=['training1', 'validation1_lab'])

    movement_interval(train_on=['training1','training2','training3', 'training4',
            'validation1_lab', 'validation2_lab', 'validation3_lab'],
        predict_on=['test1', 'test2', 'test3', 'test4', 'test5', 'test6'])
