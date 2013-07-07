
f = open('create_submission.txt', 'rb')
a = open('cpp/output_validation.csv','rb')
output = open('Submission.csv','wb', )
output.write('Id,Sequence\n')

for line in f:
    spl = line.split(' ')
    Sample_id = spl[0]
    nb_of_gestures_per_file = int(spl[1])
    output.write('%s,'%(Sample_id))
    for i in range(nb_of_gestures_per_file):
        b = a.readline()
        c = b[:-1].split(' ')
        index_high = -1
        highest = -2
        for index, j in enumerate(c):
            #print index, j
            if(highest < float(j)):
                highest = float(j)
                index_high = index+1
        if i==nb_of_gestures_per_file-1:
            output.write('%d'%(index_high))
        else:
            output.write('%d '%(index_high))
    output.write('\n')
    
f.close()
a.close()
output.close()
        
                
            
