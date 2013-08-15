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
    You need : pyplot, numpy, scipy, opencv, scikit-learn
    
    You need:
    
    - algo.py
    - Skelet.py
    - mfcc.py
    - mel.py
    - Head_interaction.py
    
    - algo_multi_modal_v3.py
    
    - levenshtein.py

    Launch algo_multi_modal_v3.py:
        It will do  #Features creation: 2159.95s = 36mn
                    #Training on gestures: 18.77s
                    #Features creation and training on sound: 15484.51s = 258mn = 4h18
                    #Full prediction: 2481.75s = 41mn
    Finish! You can now submit the newly created file 'Submission.csv'
    
    You can verify your score with levenshtein.py

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
