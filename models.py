from gesture_modelling import gesture_to_id
from postprocessing import postprocess, dump_predictions
from preprocessing import aggregated_skeletion, aggregated_skeletion_win
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from pandas import DataFrame
import numpy as np
from joblib import Memory
memory = Memory('cache/')


@memory.cache
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

    X_win = aggregated_skeletion_win(['validation1_lab', 'validation2_lab',
        'validation3_lab'], agg_functions=['median', 'var', 'min', 'max'])

    y_pred = clf.predict_proba(X_win)
    df_pred = DataFrame(y_pred, index=[s for (s, _) in X_win.index])

    to_dump = df_pred.groupby(level=0).apply(postprocess)
    dump_predictions(to_dump, out_path=out_file)
    return to_dump


@memory.cache
def eval_seq_model(out_file='eval_model.csv', retrain=False):

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
            agg_functions=['median', 'var', 'min', 'max'])

    y_pred = clf.predict_proba(X_win)
    df_pred = DataFrame(y_pred, index=[s for (s, _) in X_win.index])

    to_dump = df_pred.groupby(level=0).apply(postprocess)
    dump_predictions(to_dump, out_path=out_file)
    return to_dump


@memory.cache
def eval_gesture_model(retrain=False):

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

    X_test, y_test = aggregated_skeletion(['validation2_lab', 'training2'],
            agg_functions=['median', 'var', 'min', 'max'])
    X_test = X_test.fillna(0)
    y_pred = clf.predict_proba(X_test)
    return y_pred, y_test


if __name__ == '__main__':
    from models import leaderboard_model, eval_seq_model, eval_gesture_model

    #leaderboard_model(retrain=True)
    #eval_seq_model(retrain=True)
    eval_gesture_model()

