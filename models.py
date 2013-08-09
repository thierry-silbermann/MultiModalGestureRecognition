from gesture_modelling import gesture_to_id
from gesture_modelling import dump_predictions
from preprocessing import aggregated_skeletion
from preprocessing import aggregated_skeletion_win
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import pandas as pd
from pandas import DataFrame
import numpy as np


def leaderboard_model(out_file='leaderboard.csv', retrain=False):

    filename = 'cache/joblib/rf_leaderboard.joblib.pkl'
    file_names=['training1', 'training2', 'training3',
                                    'training4']

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

    X_win = aggregated_skeletion_win(['validation1_lab', 'validation2_lab', 'validation3_lab'],
            agg_functions=['median', 'var', 'min', 'max'])

    y_pred = clf.predict(X_win)
    df_pred = DataFrame.from_records(X_win.index.tolist(), columns=['sample_id', 'frame'])
    df_pred['gesture'] = y_pred

    dump_predictions(df_pred,out_file , convert_to_id=False)

    print 'done, predictions are in ' + out_file


if __name__ == '__main__':

    leaderboard_model(retrain=True)
