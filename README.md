Folder structure
----------------

All the .py file should be in the same directory. You need to change the root path in algo_multi_model_v3.py in the main function.
The data should be unzipped in the root, so to have the following path from the root to a .mat file: 'root/trainingX/SampleXXXXX/SampleXXXXX_data.mat
The data is unzipped on all level but it is quite easy to deal with zip file if needed.

I only supplied the source file. Everything can be generated from them. (Principally the models). If needed I can supply the models in a pickle format (around 67Mb)

The prediction can be done on individual sample or all the sample.

I deleted Sample0177 from the training file because the data looked really weird.

The library dependency are normally: numpy, scipy, scikit-learn
The code was tested on Ubuntu 13.04 with:
Python 2.7.4 / 
Numpy 1.7.1 / 
Scipy 0.11.0 / 
Scikit-learn 0.13.1

Any questions or problems? thierry.silbermann@gmail.com

#########################

Actual Model with Blending of 3 models
------------

    You need these classes (nothing need to be change in these 4 classes):
    - Head_interaction.py
    - VideoMat.py
    - Skelet.py
    - mfcc.py
    - mel.py
    
This is the main program: algo_multi_modal_v3.py (only changed should be made in the main() function)

    Launch algo_multi_modal_v3.py:
        It will do  #Features creation and training on gestures: 20mn
                    #Features creation and training on sound: 4h18
                    #Full prediction: 41mn
    Finish! You can now submit the newly created file 'Submission.csv'
    
#########################

