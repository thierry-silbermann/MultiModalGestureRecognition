#http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

import unittest

def levenshtein(seq1, seq2):
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]
    
   
def test():
    assert levenshtein([1,2,4], [3,2]) == 2
    assert levenshtein([1], [2]) == 1
    assert levenshtein([2,2,2], [2]) == 2
    print 'All tests passed'

def compute_score(filename1, filename2):
    file1 = open(filename1,"rb")
    file2 = open(filename2,"rb")
    
    line_file1 = file1.readline()
    line_file1 = file1.readline()
    line_file2 = file2.readline()
    line_file2 = file2.readline()
    
    sum_ = 0
    total = 0
    while(line_file1 != ''):
        arr_1 = line_file1[:-1].split(',')
        arr_1 = arr_1[1].split(' ')
        
        arr_2 = line_file2[:-1].split(',')
        arr_2 = arr_2[1].split(' ')
        
        total += len(arr_2)
        sum_ += levenshtein(arr_1, arr_2)
        #print arr_1, arr_2, levenshtein(arr_1, arr_2)
        
        line_file1 = file1.readline()
        line_file2 = file2.readline()
    
    return sum_, total, float(sum_)/total


if __name__ == '__main__':
    test()
    #print compute_score('training_prediction.csv', 'Submission_training.csv')
    sum_, total, score = compute_score('Submission.csv', 'validation_submission.csv') #the second file need to be the truth value file for gestures
    print score
