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
    - when training is done, launch './sound_validation > output_validation.csv'
    - We can now create the submission file with: 'algo_create_submission.py'


    Actually the submission score with these file is: 1.50654
    Worse than random...
    
    Improvement
    -----------
    Find a better interval selection approach in algo_sound.py 
    For this use information from movement

#########################

Actual Model
------------

    You need:
    - Skelet.py
    - algo.py
    - algo_move_training.py
    - algo_create_submission.py
    - sound_training.c (sound_training)
    - sound_validation.c (sound_validation)

    Before launching algo_move_training.py, remove cpp/movement_training.data and cpp/movement_validation.data
    Launch algo_move_training.py
    Make sure that there is no empty line at the end of the 'create_submission.txt'
    Verify the first line of the cpp/movement_training.data
        The first line is: 
            7107 number of interval (samples created / normally one per movement). To find the right number, use (count number of line - 1)/2
            1760 is (80('time_frame') * len(data_joints))
            20 is the number of different movement 
    Look in sound_training.c, verify that num_input is equal to the second number in the first line in movement_training.data
    Same thing for sound_validation.c
    Compile with make
    Launch the training of the neural network: ./sound_training
    Classify test by launching: ./sound_validation > output_validation.csv
    Now everything is ready to create the submission file, you need:
        - create_submission.txt
        - output_validation.csv
    Launch algo_create_submission.py
    Finish! You can now submit the newly created file 'Submission.csv'
    
    
            
    
