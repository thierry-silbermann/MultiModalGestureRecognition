TODO
----
create json file with local path to avoid conflict when push


folder structure
----------------

put the original dat in:
data/raw_data/

#########################

Actual Model with Blending of 3 models
------------
    You need : numpy, scipy, opencv, scikit-learn
    
    You need:
    
    - VideoMat.py
    - Skelet.py
    - mfcc.py
    - mel.py
    
    - algo_multi_modal_v3.py

    Launch algo_multi_modal_v3.py:
        It will do  #Features creation and training on gestures: 20mn
                    #Features creation and training on sound: 4h18
                    #Full prediction: 41mn
    Finish! You can now submit the newly created file 'Submission.csv'
    
#########################

Bad samples
-------

Sample00011 15
Sample00014 15
Sample00071 8
Sample00072 11
Sample00173 12
Sample00174 12
Sample00175 13
Sample00176 12
Sample00177 15
Sample00179 13
Sample00181 14
Sample00183 15
Sample00187 14
Sample00338 10
Sample00339 14
Sample00374 16
Sample00391 14
