#include "extra.h"

float * gaussianPatch(const int size, const float sigma){
    
    float *gauss = (float*)malloc(size*size*(sizeof(float)));
    float sum = 0;
    float max = 0;
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            float x = i - (size - 1) / 2.0;
            float y = j - (size - 1) / 2.0;
            gauss[i*size + j] =  exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(sigma, 2)))) * (-1));
            sum += gauss[i*size + j];
        }
    }
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            gauss[i*size + j] /= sum;
            if(gauss[i*size + j]>max){
                max=gauss[i*size + j];
            }
        }
    }
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            gauss[i*size + j] /= max;
        }
    }
    return gauss;
}