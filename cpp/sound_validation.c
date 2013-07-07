#include <string.h>
#include <stdio.h>
#include "floatfann.h"

int main()
{
    int data_size = 16000; 
    fann_type *calc_out;
    fann_type input[data_size];

    struct fann *ann = fann_create_from_file("sound_float.net"); 

    static const char filename[] = "sound_validation.data";
    FILE *file = fopen ( filename, "r" );
    if ( file != NULL ) {
        char line [ 100000 ]; /* or other suitable maximum line size */
        while ( fgets ( line, sizeof line, file ) != NULL ) /* read a line */ {
            //fputs ( line, stdout ); /* write the line */
            
            char * p    = strtok (line, " ");
            int n_spaces = 0;

            // split string and append tokens to 'input' 
            while(p) {
                input[n_spaces] = strtod(p, NULL);
                n_spaces++;
                p = strtok(NULL, " ");
            }
            
            calc_out = fann_run(ann, input);

            printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", 
            calc_out[0], calc_out[1], calc_out[2], calc_out[3], calc_out[4]
            , calc_out[5], calc_out[6], calc_out[7], calc_out[8], calc_out[9]
            , calc_out[10], calc_out[11], calc_out[12], calc_out[13], calc_out[14]
            , calc_out[15], calc_out[16], calc_out[17], calc_out[18], calc_out[19]);
            
        }
        fclose ( file );
    } else {
        perror ( filename ); /* why didn't the file open? */
    }

    fann_destroy(ann);
    return 0;
}
