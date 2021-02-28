#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "extra.h"

//#define bAbs(a,b,c,d)   ((a+b)<(c)?(c-(a+b)-1):((a+b)>(d)?(2*d-(a+b)+1):(a+b))) //bounded absolute
//#define bAbs(a,b,c)   ((a)<(b)?(b-a):(((a)>(c)?(2*c-a)))

int bAbs(const int a, const int b, const int c){
    int temp;
    if(a<b){
        temp = 2*b-a-1;
        return temp;
    }else{
        if(a>c){
            temp = 2*c-a+1;
            return temp;
        }else{
            return a;
        }
    }
}
void print_arr(float* arr, int N, int M){
    printf("\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            printf("[%2.0f]",arr[i*M + j]);
        }
        printf("\n");
    }
}

void patchCube(float* input, float*output, int N, int M, int window, const float pathcSigma){
    
    int windSize=window*window;
    int halfWind=(window-1)/2;

    float * gauss = gaussianPatch(window, pathcSigma);
    

    //print_arr(gauss,window,window);

    for(int j=0; j<M; j++){
        for(int i=0; i<N; i++){
            for(int kx=-halfWind; kx<=halfWind; kx++){
                for(int ky=-halfWind; ky<=halfWind; ky++){
                    output[i*M*windSize + j*windSize + (kx+halfWind)*window +(ky+halfWind)] = gauss[(ky+halfWind)*window +(kx+halfWind)] * input[bAbs(i+ky,0,N-1)*M +bAbs(j+kx,0,M-1)];
                }
            }
                /* printf("\n"); */   
        }
    }

    free(gauss);
}
void patchCube2(float* input, float*output, int N, int M, int window, const float pathcSigma){
    
    int windSize=window*window;
    int halfWind=(window-1)/2;

    float * gauss = gaussianPatch(window, pathcSigma);
    

    //print_arr(gauss,window,window);

    for(int j=0; j<M; j++){
        for(int i=0; i<N; i++){
            for(int kx=-halfWind; kx<=halfWind; kx++){
                for(int ky=-halfWind; ky<=halfWind; ky++){
                    output[j*M*windSize + i*windSize + (kx+halfWind)*window +(ky+halfWind)] = gauss[(ky+halfWind)*window +(kx+halfWind)] * input[bAbs(i+ky,0,N-1)*M +bAbs(j+kx,0,M-1)];
                }
            }  
        }
    }

    free(gauss);
}
void nlm(float * out, const float * in, const float *cube,const int N, const int M, const int window, const float filtSigma){
    
    int winSize =  window*window;
    float D;
    float sum  = 0;
    float Dsum = 0;
    float maxD = 0;
    float temp = 0;

    for(int i=0; i<N*M; i++){
        for(int j=0; j<N*M; j++){

            for(int k=0; k<winSize; k++){
                temp = cube[i*winSize + k]-cube[j*winSize + k];
                sum += temp*temp;
            }

            D = exp(-sum/filtSigma);
            
            if(D!=1){
                Dsum += D;
                out[i] += D*in[j];
                if( D > maxD){
                    maxD = D;
                }
            } 
            
            sum   = 0;

        }
        //make the diagonal element be the max (line 56 matlab)
        out[i] += maxD*in[i];
        Dsum   += maxD;
        maxD    = 0;

        out[i] /= Dsum;
        Dsum    = 0;
    }

}


int main(int argc, char const *argv[])
{
    srand(1);

    FILE * png =       fopen(argv[1],"r");
    int N =             atoi(argv[2]);
    int M =             atoi(argv[3]);
    int window =        atoi(argv[4]);
    float patchSigma =  atof(argv[5]);
    float filtSigma =   atof(argv[6]);

    
    if( png == NULL ){
        printf("Couldn't load png\n");
        exit(1);
    }


    float * data     =  malloc(N*M*sizeof(float));
    float * filtered =  malloc(N*M*sizeof(float));
    float * padded   =  malloc(N*M*window*window*sizeof(float));



    for( int i=0; i<N*M; i++){
        if(!fscanf(png,"%f,",&data[i])) break;
    }
    fclose(png);

    //print_arr(data,N,M);
    
    patchCube(data,padded,N,M,window,patchSigma);

    //print_arr(padded,N*M,window*window);

    nlm(filtered,data,padded,N,M,window,filtSigma);

    //print_arr(filtered,N,M);

    FILE * filtPng = NULL;
    filtPng = fopen("filtPng.csv","w");
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            if(j!=M-1)
                fprintf(filtPng,"%.16f,",filtered[i*M + j]);
            else
                fprintf(filtPng,"%.16f" ,filtered[i*M + j]);
        }
        fprintf(filtPng,"\n");
    }
    fclose(filtPng);

    free(data);
    free(padded);
    free(filtered);

    return 0;
}

int main1(int argc, char const *argv[])
{
    float *ttt = malloc(25*sizeof(float));
    for(int i=0; i<25; i++){
        ttt[i] = i;
    }
    float *ddd = malloc(25*9*sizeof(float));

    //printf("hi\n");
    patchCube(ttt,ddd,5,5,3,5/3);

    print_arr(ddd,25,9);
    return 0;
}
