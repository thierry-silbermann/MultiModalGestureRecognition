I only supply the source file. Everything can be generated from them.
(Principally the models). If needed I can supply the models in a pickle format
The prediction can be done on individual sample or all the sample.


in order to run the model you need to:
=====================================

1. You need to change the root path in 'algo_multi_model_v3.py' in the main function.
2. extract and make some folders (see Folder structure and manual data preparation)
3. install the dependencies
4. issue on the terminal: 'python run_models.py' inside the root folder


Folder structure and manual data preparation
---------------------------------------------

1. All the .py file should be in the same directory.
2. The data should be unzipped in the root, so to have the following path from the root
    to a .mat file: 'root/trainingX/SampleXXXXX/SampleXXXXX_data.mat
3. The data is unzipped on all level but it is quite easy to deal with zip file if needed.
4. I deleted Sample0177 from the training file because the data looked really weird.
5. create folder 'root/cache'
6. in addition to the extracted data files, all the files (training, validation_lab, test)
    have also to be in 'root/data/raw_data' in there orignal format '.gz'

DEPENDENCIES:
------------
The library dependencies are normally: numpy, scipy, scikit-learn
The code was tested on Ubuntu 13.04 with:

joblib==0.7.0d
matplotlib==1.2.1
pandas==0.11.0
scipy==0.12.0

Python 2.7.4
Numpy 1.7.1
Scipy 0.11.0
Scikit-learn 0.13.1

Any questions or problems? thierry.silbermann@gmail.com or 
immanuel.bayer@uni-konstanz.de

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

