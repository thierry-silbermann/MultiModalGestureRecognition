
f = open('create_submission2.txt', 'rb') #create_submission.txt
a = open('cpp/output_validation2.csv','rb') #cpp/output_validation.csv
output = open('Submission2.csv','wb', ) #Submission.csv
output.write('Id,Sequence\n')

threshold = 0.7
for line in f:
    spl = line.split(' ')
    Sample_id = spl[0]
    nb_of_gestures_per_file = int(spl[1])
    output.write('%s,'%(Sample_id))
    actual_gesture = -1
    for i in range(nb_of_gestures_per_file):
        b = a.readline()
        c = b[:-1].split(' ')
        index_high = -1
        highest = -2
        for index, j in enumerate(c):
            if(highest < float(j)):
                highest = float(j)
                index_high = index+1
        
        if (highest > threshold):   # Only keep high probability match
            if actual_gesture != index_high:    #No repetition
                actual_gesture = index_high
                if i==nb_of_gestures_per_file-1:
                    output.write('%d'%(index_high))
                else:
                    output.write('%d '%(index_high))
    output.write('\n')
    
f.close()
a.close()
output.close()
        
                
            
