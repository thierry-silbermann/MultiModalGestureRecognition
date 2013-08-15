I only supply the source file. Everything can be generated from them.
(Principally the models). If needed I can supply the models in a pickle format
The prediction can be done on individual sample or all the sample.


########## Method 1 ###############

The code was tested on Ubuntu 13.04 with:
The library dependencies are normally

- openCV 2.4.2 (shouldn't need it in fact)
- Python 2.7.4
- Numpy 1.7.1
- Scipy 0.11.0
- Scikit-learn 0.13.1

Actual Model with Blending of 3 models
------------

    You need these classes (nothing need to be change in these 4 classes):
    - Head_interaction.py
    - VideoMat.py
    - Skelet.py
    - mfcc.py
    - mel.py


1. install the dependencies
2. All the .py file should be in the same directory.
3. The data should be unzipped in the root, so to have the following path from the root
    to a .mat file: 'root/trainingX/SampleXXXXX/SampleXXXXX_data.mat
4. The data is unzipped on all level but it is quite easy to deal with zip file if needed.
5. I deleted Sample0177 from the training file because the data looked really weird.
6. You need to change the root path in 'algo_multi_model_v3.py' in the main function.

This is the main program: algo_multi_modal_v3.py (only changed should be made in the main() function)

    Launch algo_multi_modal_v3.py:
        It will do  #Features creation and training on gestures: 20mn
                    #Features creation and training on sound: 4h18
                    #Full prediction: 41mn
    Finish! You can now submit the newly created file 'Submission.csv'
    This submission is not our best model. It will normally give a score of 0.33
 
Any questions or problems? thierry.silbermann@gmail.com   

######### Method 2 ##############

in order to run the model you need to:
=====================================


Folder structure and manual data preparation
---------------------------------------------

1. All the .py file should be in the same directory.
2. create folder 'root/cache'
3. in addition to the extracted data files, all the files (training, validation_lab, test)
    have also to be in 'root/data/raw_data' in there orignal format '.gz'

DEPENDENCIES:
------------
The code was tested on Ubuntu 13.04 with:
The library dependencies are normally

- joblib==0.7.0d
- matplotlib==1.2.1
- pandas==0.11.0
- scipy==0.12.0

Any questions or problems? immanuel.bayer@uni-konstanz.de


######## Best Score #################


Our current best score: '0.2596' is a blending from the result of the two different methods.
The code is ready but we still testing it to be sure that it can be deployed easily by launching: 'python run_models.py' inside the root folder


