from preprocessing import aggregated_skeletion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.externals import joblib
import pandas as pd
from functools import partial


def trained_rf(file_names=['training1', 'training2', 'training3',
                                    'training4'], recompute=False):
    filename = 'cache/joblib/gesture_rf.joblib.pkl'

    if recompute:
        X, y = aggregated_skeletion(file_names=file_names,
                agg_functions=['median', 'var'])
        X = X.fillna(0)

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)


        clf = ExtraTreesClassifier(n_estimators=500, random_state=0,
            n_jobs=-1)
        clf.fit(X, y)

        _ = joblib.dump((clf, le), filename, compress=9)
    else:
        clf, le = joblib.load(filename)

    return clf, le
