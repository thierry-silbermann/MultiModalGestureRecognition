from preprocessing import aggregated_skeletion
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import pandas as pd
import numpy as np



gesture_to_id = {'vieni':2, 'break':0, 'prend':14, 'sonos':20, 'chevu':6,
        'dacco':7, 'perfe':3, 'vatte':1, 'basta':13, 'buoni':18, 'chedu':5,
        'cosat':12, 'fame':16, 'nonce':15, 'furbo':4, 'combi':9, 'frega':10,
        'seipa':8, 'tanto':17, 'messi':19, 'ok':11}
id_to_gesture = dict(zip(gesture_to_id.values(), gesture_to_id.keys()))


def trained_rf(file_names=['training1', 'training2', 'training3',
                                    'training4'], recompute=False):
    filename = 'cache/joblib/gesture_rf.joblib.pkl'

    if recompute:
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

    return clf


def dump_predictions(df, out_path='pred.csv', convert_to_id=True):

    grouped = df.groupby('sample_id')

    with open(out_path, 'w') as f:
        f.write('Id,Sequence\n')
        for sample_id, group in grouped:
            gestures = group.sort('frame').drop_duplicates('gesture')
            if convert_to_id:
                gestures = ' '.join([str(gesture_to_id[gest]) for gest in gestures.gesture if gesture_to_id[gest] > 0])
            else:
                gestures = ' '.join([str(gest) for gest in gestures.gesture if gest > 0])

            out = sample_id[-4:] + ',' + gestures + '\n'
            f.write(out)
