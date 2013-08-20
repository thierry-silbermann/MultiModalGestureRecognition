import pandas as pd
from pandas import DataFrame
import numpy as np

gesture_to_id = {'vieni':2, 'break':0, 'prend':14, 'sonos':20, 'chevu':6,
        'dacco':7, 'perfe':3, 'vatte':1, 'basta':13, 'buoni':18, 'chedu':5,
        'cosat':12, 'fame':16, 'nonce':15, 'furbo':4, 'combi':9, 'frega':10,
        'seipa':8, 'tanto':17, 'messi':19, 'ok':11}
id_to_gesture = dict(zip(gesture_to_id.values(), gesture_to_id.keys()))


def postprocess(scan_line, remove_below_break=True, remove_break=True,
        min_dist_between_max=5, min_proba=0.05):

    #assumptions used:
    # 1. if break proba is hight, there is no gesture
    # 2. each gesture can only appear once
    # 3. min dist between gestures
    # 4. min proba !
    scan_line = scan_line.values
    records = []
    occupied_frame_pos = []

    available_gestures = set(range(1, scan_line.shape[1]))


    for i in xrange(scan_line.shape[0]):
        if remove_below_break:
            i_smaller_break = scan_line[i, :] < scan_line[i, 0]
            scan_line[i, i_smaller_break] = 0.0

    if remove_break:
        scan_line[:, 0] = 0.0

    tmp = scan_line.copy()

    for gest in range(scan_line.shape[1]):
        #print available_gestures

        max_frame, max_gesture = np.unravel_index(np.argmax(tmp), scan_line.shape)
        max_proba = scan_line[max_frame, max_gesture]
        if occupied_frame_pos:
            min_dist = np.min(np.abs(np.array(occupied_frame_pos) -
                float(max_frame)))
        else: 
            min_dist = min_dist_between_max + 1

        if max_gesture in available_gestures and min_dist > min_dist_between_max and max_proba > min_proba:

            # delete right
            curr_frame = max_frame
            curr_proba = max_proba
            while curr_proba > 0 and curr_frame + 1 < scan_line.shape[0] and np.argmax(scan_line[curr_frame, :]) == max_gesture:
                curr_proba = scan_line[curr_frame, max_gesture]
                scan_line[curr_frame, :] = 0.0
                tmp[curr_frame, :] = 0.0
                scan_line[curr_frame, max_gesture] = curr_proba
                curr_frame += 1

            # delete left
            curr_frame = max_frame
            curr_proba = scan_line[max_frame, max_gesture]
            while curr_proba > 0 and curr_frame > 0 and np.argmax(scan_line[curr_frame, :]) == max_gesture:
                curr_proba = scan_line[curr_frame, max_gesture]
                scan_line[curr_frame, :] = 0.0
                tmp[curr_frame, :] = 0.0
                scan_line[curr_frame, max_gesture] = curr_proba
                curr_frame -= 1

            available_gestures.remove(max_gesture)
            records.append((max_gesture, max_frame, scan_line[max_frame, max_gesture]))
            occupied_frame_pos.append(max_frame)
            tmp[:, max_gesture] = 0.0
        # delete so that next max can be found
        else:
            # delete right
            curr_frame = max_frame
            while curr_proba > 0 and curr_frame + 1 < scan_line.shape[0]:
                tmp[curr_frame, max_gesture] = 0.0
                curr_frame += 1

            # delete left
            curr_frame = max_frame
            while curr_proba > 0 and curr_frame > 0:
                tmp[curr_frame, max_gesture] = 0.0
                curr_frame -= 1

    return DataFrame.from_records(records,
            columns=['gesture', 'frame', 'p']).sort('frame')


def dump_predictions(df, out_path='pred.csv', convert_to_id=False):

    grouped = df.groupby(level=0)

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


# source: SciPy-CookBook
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[(window_len/2 - (window_len + 1)%2):-(window_len/2)]


def pad_smooth(sample, window_len):
    nr_frames = sample.frame.max() + 40
    out = DataFrame({'frame': np.arange(nr_frames)})
    out['sample_id'] = sample.sample_id.unique()[0]

    out = pd.merge(out, sample, how='outer', on=['sample_id', 'frame'])
    out.fillna(method='ffill', inplace=True, limit=2)
    out.fillna(method='bfill', inplace=True)
    out.fillna(method='ffill', inplace=True)
    for gesture in range(21):
        out[gesture] = smooth(out[gesture], window_len=window_len, window='hanning')
    return out
