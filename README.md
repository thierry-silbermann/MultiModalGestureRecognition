TODO
----
create json file with local path to avoid conflict when push


folder structure
----------------

put the original dat in:
data/raw_data/

Mouvement part
----------
    For now the program only show plot of joint movement. 
    The red and green line represent the beginning and ending (respectively) given by
    the data in the .mat file for each sample.

    You need:
    algo.py
    algo_move.py
    
    - Change path in function getAllWav()
    - launch 'python algo_move.py'

sound part
----------

    You need:
    algo.py
    algo_sound.py
    algo_create_submission.py
    
    install FANN library in C: http://leenissen.dk/fann/wp/download/
    and all the file in cpp/
    
    - Change path in function getAllWav() and main()
    - verify that there is no 'sound_validation.data' file in cpp/
    - launch 'python algo_sound.py'
    - After compiling file in cpp/ with 'make'
    - launch ./sound_training
    - when training is done, launch ./sound_validation
    - We can now create the submission file with: 'algo_create_submission.py'


    Actually the submission score with these file is: 1.50654
    Worse than random...
    
    Improvement
    -----------
    Find a better interval selection approach in algo_sound.py 
    For this use information from movement


