#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "extra.h"
#include "gputimer.h"
#include "sort.h"


// ONLY nlmSimple AND nlmShared ARE COMMENTED


// prints N by M array
void print_arr(const float* arr, int N, int M){
    printf("\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            printf("[%0.3f]",arr[i*M + j]);
        }
        printf("\n");
    }
}

// nlm algorithm using global memory
__global__ void nlmSimple(float *out, const float *in, const float *cube,
                          const int N, const int M, const int window,
                          const float filtSigma){

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    const int winSize = window*window;
    const int picSize = N*M;
    float sum  = 0;
    float Dsum = 0;
    float maxD = 0;
    float tempOut = 0;
    float D;

    if( i >= N*M) return; //check to see if we want the thread to do work

    float temp;
    // main loop
    for(int j=0; j<picSize; j++){
        if(i==j) continue; //check to see if we are calculating distance to ourselves

        #pragma unroll 4
        for(int k=0; k<winSize; k++){ //calculate distance and sum up the squared value
            temp = cube[i*winSize + k]-cube[j*winSize + k];
            sum += temp*temp;
        }

        D = expf(-sum/filtSigma);

        Dsum+= D;
        tempOut += D*in[j];
        if(D > maxD){   //we need the max for each pixel to recreate
            maxD = D;   //line 56 in the matlab pipeline
        }

        sum   = 0;

    }
    //make the diagonal element be the max (line 56 matlab)
    tempOut += maxD*in[i];
    Dsum    += maxD;

    out[i]   = tempOut/Dsum;

}

// nlm algorithm using global memory but with the transpose of *cube,
// making accesses coalesced
__global__ void nlmSimpleT(float *out, const float *in, const float *cube,
                          const int N, const int M, const int window,
                          const float filtSigma){

    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    const int winSize = window*window;
    const int picSize = N*M;
    float sum  = 0;
    float Dsum = 0;
    float maxD = 0;
    float tempOut = 0;
    float D;

    if( i >= N*M) return;

    float temp;

    for(int j=0; j<picSize; j++){

        if(i==j) continue;

        //#pragma unroll 8
        for(int k=0; k<winSize; k++){
            temp = cube[k*picSize + i]-cube[k*picSize + j];
            sum += temp*temp;
        }

        D = expf(-sum/filtSigma);

        Dsum += D;
        tempOut += D*in[j];
        if( D > maxD){
            maxD = D;
        }

        sum   = 0;

    }
    //make the diagonal element be the max (line 56 matlab)
    // D[i] (diagonal): sum=0 -> e^0=1 -> D=1
    tempOut += maxD*in[i];
    Dsum    += maxD;

    out[i]   = tempOut/Dsum;

}

// nlm algorithm using shared memory
// using strip mining, does work in blocks,
// bringing in shared memory only the data it need every time
__global__ void nlmShared(float *out, const float *in, const float *cube,
                          const int N, const int M, const int window,
                          const float filtSigma){

    const int blockSize = blockDim.x;
    const int tid = threadIdx.x;
    const int i = blockSize * blockIdx.x + tid; // which pixel, row major
    const int winSize = window*window;

    float sum  = 0;
    float Dsum = 0;
    float maxD = 0;
    float tempOut = 0;
    float D;

    extern __shared__ float sh_mem[]; //shared memory, defined on call

    //break shared memory to pieces
    float *sh_in        = &sh_mem[0]; //input array
    float *sh_cubeSelf  = &sh_mem[blockSize]; //patchCube array for the values of the threads pixel
    float *sh_cubeElse  = &sh_mem[blockSize + blockSize*winSize]; //patchCube array for the values of the pixel in the current blocl

    if( i >= N*M) return;

    const float inI = in[i]; // the value of the thread pixel

    //bring in the self patchCube
    for(int cubeLine=0; cubeLine < winSize; cubeLine++) {
        sh_cubeSelf[tid*winSize + cubeLine] = cube[i*winSize + cubeLine];
    }

    int inIndex;
    float temp;

    //blocking the algorithm into N*M/blockSize pieces
    for(int p=0; p<N*M/blockSize; p++){

        //inIndex = (i+p*blockSize)%(N*M); // mod % to cycle arround to the start
        inIndex = tid+p*blockSize; // this also works, every block starts from the same point

        // sync before writing to make sure everyone has read
        __syncthreads();
        sh_in[tid] = in[inIndex]; //bringing in the current block's pixel value

        //bringing in the current block's patchCube
        #pragma unroll 8
        for(int cubeLine=0; cubeLine < winSize; cubeLine++) {
            sh_cubeElse[tid*winSize + cubeLine] = cube[inIndex*winSize + cubeLine];
        }
        // sync before reading to make sure everyone has writen
        __syncthreads();

        for(int j=0; j<blockSize; j++){

            for(int k=0; k<winSize; k++){
                temp = sh_cubeSelf[tid*winSize + k]-sh_cubeElse[j*winSize + k];
                sum += temp*temp;
            }

            D = expf(-sum/filtSigma);

            //if we are calculating distance to ourselves, sum=0 -> D=e^0=1
            if(D!=1){
                Dsum += D;
                tempOut += D*sh_in[j];
                if( D > maxD){
                    maxD = D;
                }
            }

            sum   = 0;

        }

    }
    tempOut += maxD*inI;
    Dsum    += maxD;

    out[i]  = tempOut/Dsum;

}

// nlm algorithm using shared memory
// also uses transpose *cube
// furthermore uses transpose shared array
// if blockSize=16 or 32 then avoids bank conflicts
__global__ void nlmSharedT(float *out, const float *in, const float *cube,
                          const int N, const int M, const int window,
                          const float filtSigma){

    const int blockSize = blockDim.x;
    const int tid       = threadIdx.x;
    const int i = blockSize * blockIdx.x + tid; // which pixel, row major
    const int winSize = window*window;
    const int picSize = N*M;

    float sum  = 0;
    float Dsum = 0;
    float maxD = 0;
    float tempOut = 0;
    float D;

    extern __shared__ float sh_mem[];

    float *sh_in        = &sh_mem[0];
    float *sh_cubeSelf  = &sh_mem[blockSize];
    float *sh_cubeElse  = &sh_mem[blockSize+ blockSize*winSize];

    if( i >= N*M) return;

    const float inI = in[i];

    for(int cubeLine=0; cubeLine < winSize; cubeLine++){
        sh_cubeSelf[cubeLine*blockSize + tid] = cube[cubeLine*picSize + i];
    }

    int inIndex;
    float temp;

    for(int p=0; p<picSize/blockSize; p++){

        //inIndex = (i+p*blockSize)%(picSize); // mod % to cycle arround to the start
        inIndex = tid+p*blockSize; // this also works, every block starts from the same point

        // sync before writing to make sure everyone has read
        __syncthreads();
        sh_in[tid] = in[inIndex];

        //#pragma unroll 8
        for(int cubeLine=0; cubeLine < winSize; cubeLine++){
            sh_cubeElse[cubeLine*blockSize + tid] = cube[cubeLine*picSize + inIndex];
        }
        // sync before reading to make sure everyone has writen
        __syncthreads();

        for(int j=0; j<blockSize; j++){
            #pragma unroll 8
            for(int k=0; k<winSize; k++){
                temp = sh_cubeSelf[k*blockSize + tid]-sh_cubeElse[k*blockSize + j];
                sum += temp*temp;
            }

            D = expf(-sum/filtSigma);

            if(D!=1){
                Dsum += D;
                tempOut += D*sh_in[j];
                if( D > maxD){
                    maxD = D;
                }
            }

            sum   = 0;

        }

    }
    tempOut += maxD*inI;
    Dsum    += maxD;

    out[i]   = tempOut/Dsum;

}

//for testing purposes, max elementwise change between matrices
float maxError(float *arr1, float*arr2, int N, int M){
    float maxE =0;
    float temp;
    for(int i=0; i<N*M; i++){
        temp = fabs(arr1[i]-arr2[i]);
        if( temp > maxE){
            maxE = temp;
        }
    }
    return maxE;
}

// bounded absolute for mirroring in padding
// assumes a is within range of the first mirroring ((window-1)/2 < N)
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

// creates the patchCube array
void patchCube(const float* input, float*output, const int N, const int M,
               const int window, const float pathcSigma){

    const int windSize=window*window;
    const int halfWind=(window-1)/2;

    float * gauss = gaussianPatch(window, pathcSigma);

    for(int j=0; j<M; j++){
        for(int i=0; i<N; i++){
            for(int kx=-halfWind; kx<=halfWind; kx++){
                for(int ky=-halfWind; ky<=halfWind; ky++){
                    output[i*M*windSize + j*windSize + (kx+halfWind)*window +(ky+halfWind)] = gauss[(ky+halfWind)*window +(kx+halfWind)] * input[bAbs(i+ky,0,N-1)*M + bAbs(j+kx,0,M-1)];
                }
            }
        }
    }

    free(gauss);
}
// benchmarking
void randTest(char const *argv[])
{
    GpuTimer timer;

    if(atoi(argv[1])==0){
        srand(time(NULL));
    }else{
        srand(atoi(argv[1]));
    }

    int runSimple  = argv[2][0]-'0';
    int runSimpleT = argv[2][1]-'0';
    int runShared  = argv[2][2]-'0';
    int runSharedT = argv[2][3]-'0';

    int N =             atoi(argv[3]);
    int M =             atoi(argv[4]);
    int window =        atoi(argv[5]);
    float patchSigma =  atof(argv[6]);
    float filtSigma =   atof(argv[7]);
    int blockSize =     atoi(argv[8]);
    int timeToRun =     atoi(argv[9]);
    int printStats =    atoi(argv[10]);

    float *timesOfSimple  = (float*)malloc(timeToRun*sizeof(float));
    float *timesOfSimpleT = (float*)malloc(timeToRun*sizeof(float));
    float *timesOfShared  = (float*)malloc(timeToRun*sizeof(float));
    float *timesOfSharedT = (float*)malloc(timeToRun*sizeof(float));

    int maxThreads, device, maxMem;
    int sharedMem = (blockSize + 2*blockSize*window*window)*sizeof(float);

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&maxThreads,cudaDevAttrMaxThreadsPerBlock,device);
    //printf("device=%d, maxthreads=%d\n",device,maxThreads);

    cudaDeviceGetAttribute(&maxMem,cudaDevAttrMaxSharedMemoryPerBlock,device);
    //printf("maxMem: %d, shared needs: %d, can run: %d warps\n",maxMem,sharedMem,maxMem/sharedMem);

    float * data     =  (float*)malloc(N*M*sizeof(float));
    float * filtered =  (float*)malloc(N*M*sizeof(float));
    float * padded   =  (float*)malloc(N*M*window*window*sizeof(float));
    float * paddedT  =  (float*)malloc(N*M*window*window*sizeof(float));


    float *d_data, *d_padded, *d_paddedT, *d_filtered, *d_filtered_sh, *d_filteredT, *d_filtered_shT;
    cudaMalloc((void **)&d_data,N*M*sizeof(float));
    cudaMalloc((void **)&d_padded,N*M*window*window*sizeof(float));
    cudaMalloc((void **)&d_paddedT,N*M*window*window*sizeof(float));
    cudaMalloc((void **)&d_filtered,N*M*sizeof(float));
    cudaMalloc((void **)&d_filtered_sh,N*M*sizeof(float));
    cudaMalloc((void **)&d_filteredT,N*M*sizeof(float));
    cudaMalloc((void **)&d_filtered_shT,N*M*sizeof(float));

    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            data[i*M + j]=(float)rand()/RAND_MAX;
        }
    }

    cudaMemcpy(d_data,data,N*M*sizeof(float),cudaMemcpyHostToDevice);

    patchCube(data,padded,N,M,window,patchSigma);

    for(int i=0; i<N*M; i++){
        for(int j=0; j<window*window; j++){
            paddedT[j*N*M +i] = padded[i*window*window + j];
        }
    }

    cudaMemcpy(d_padded,padded,N*M*window*window*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_paddedT,paddedT,N*M*window*window*sizeof(float),cudaMemcpyHostToDevice);

    int numOfBlocks = ceil((float)N*M/blockSize);
    //printf(" N*M=%d, b*s=%d\n\n",N*M,numOfBlocks*blockSize);

    for(int i=0; i<timeToRun; i++){
        // simple
        if(runSimple==1){
            timer.Start();
            nlmSimple<<<numOfBlocks,blockSize>>>(d_filtered,d_data,d_padded,N,M,window,filtSigma);
            timer.Stop();

            cudaMemcpy(filtered,d_filtered,N*M*sizeof(float),cudaMemcpyDeviceToHost);

            //printf("simple: %g ms, dif=%f\n",timer.Elapsed(),maxError(data,filtered,N,M));
            printf("simple,  %d, %d, %d, %f, 1\n",N,window,blockSize,timer.Elapsed()/1000);
            timesOfSimple[i] = timer.Elapsed()/1000;
        }


        // simple Transpose
        if(runSimpleT==1){
            timer.Start();
            nlmSimpleT<<<numOfBlocks,blockSize>>>(d_filteredT,d_data,d_paddedT,N,M,window,filtSigma);
            timer.Stop();

            cudaMemcpy(filtered,d_filteredT,N*M*sizeof(float),cudaMemcpyDeviceToHost);

            //printf("simpleT: %g ms, dif=%f\n",timer.Elapsed(),maxError(data,filtered,N,M));
            printf("simpleT, %d, %d, %d, %f, 1\n",N,window,blockSize,timer.Elapsed()/1000);
            timesOfSimpleT[i] = timer.Elapsed()/1000;
        }

        // shared
        if(runShared==1){
            timer.Start();
            nlmShared<<<numOfBlocks,blockSize,sharedMem>>>(d_filtered_sh,d_data,d_padded,N,M,window,filtSigma);
            timer.Stop();
            //print_arr(data,N,M);

            cudaMemcpy(filtered,d_filtered_sh,N*M*sizeof(float),cudaMemcpyDeviceToHost);

            //printf("shared: %g ms, dif=%f\n",timer.Elapsed(),maxError(data,filtered,N,M));
            printf("shared,  %d, %d, %d, %f, %d\n",N,window,blockSize,timer.Elapsed()/1000,maxMem/sharedMem);
            timesOfShared[i] = timer.Elapsed()/1000;
        }

        // shared Transpose
        if(runSharedT==1){
            timer.Start();
            nlmSharedT<<<numOfBlocks,blockSize,sharedMem>>>(d_filtered_shT,d_data,d_paddedT,N,M,window,filtSigma);
            timer.Stop();

            cudaMemcpy(filtered,d_filtered_shT,N*M*sizeof(float),cudaMemcpyDeviceToHost);

            //printf("sharedT: %g ms, dif=%f\n",timer.Elapsed(),maxError(data,filtered,N,M));
            printf("sharedT, %d, %d, %d, %f, %d\n",N,window,blockSize,timer.Elapsed()/1000,maxMem/sharedMem);
            timesOfSharedT[i] = timer.Elapsed()/1000;
        }
    }

    if(printStats==1){
        printf("\nMedian:\n");

        if(runSimple==1){
            quickSort(timesOfSimple,0,timeToRun-1);
            printf("simple,  %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimple[(timeToRun-1)/2]);
        }
        if(runSimpleT==1){
            quickSort(timesOfSimpleT,0,timeToRun-1);
            printf("simpleT, %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimpleT[(timeToRun-1)/2]);
        }
        if(runShared==1){
            quickSort(timesOfShared,0,timeToRun-1);
            printf("shared,  %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfShared[(timeToRun-1)/2],maxMem/sharedMem);
        }
        if(runSharedT==1){
            quickSort(timesOfSharedT,0,timeToRun-1);
            printf("sharedT, %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfSharedT[(timeToRun-1)/2],maxMem/sharedMem);
        }


        float sum = 0;

        printf("\nMean:\n");

        if(runSimple==1){
            for(int i=0; i<timeToRun; i++){
                sum += timesOfSimple[i];
            }
            printf("simple,  %d, %d, %d, %f, 1\n",N,window,blockSize,sum/timeToRun);
            sum=0;
        }
        if(runSimpleT==1){
            for(int i=0; i<timeToRun; i++){
                sum += timesOfSimpleT[i];
            }
            printf("simpleT, %d, %d, %d, %f, 1\n",N,window,blockSize,sum/timeToRun);
            sum=0;
        }
        if(runShared==1){
            for(int i=0; i<timeToRun; i++){
                sum += timesOfShared[i];
            }
            printf("shared,  %d, %d, %d, %f, %d\n",N,window,blockSize,sum/timeToRun,maxMem/sharedMem);
            sum=0;
        }
        if(runSharedT==1){
            for(int i=0; i<timeToRun; i++){
                sum += timesOfSharedT[i];
            }
            printf("sharedT, %d, %d, %d, %f, %d\n",N,window,blockSize,sum/timeToRun,maxMem/sharedMem);
            sum=0;
        }

        printf("\nMinimum:\n");

        if(runSimple==1){
            printf("simple,  %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimple[0]);
        }
        if(runSimpleT==1){
            printf("simpleT, %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimpleT[0]);
        }
        if(runShared==1){
            printf("shared,  %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfShared[0],maxMem/sharedMem);
        }
        if(runSharedT==1){
            printf("sharedT, %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfSharedT[0],maxMem/sharedMem);
        }

        printf("\nMaximum:\n");

        if(runSimple==1){
            printf("simple,  %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimple[timeToRun-1]);
        }
        if(runSimpleT==1){
            printf("simpleT, %d, %d, %d, %f, 1\n",N,window,blockSize,timesOfSimpleT[timeToRun-1]);
        }
        if(runShared==1){
            printf("shared,  %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfShared[timeToRun-1],maxMem/sharedMem);
        }
        if(runSharedT==1){
            printf("sharedT, %d, %d, %d, %f, %d\n",N,window,blockSize,timesOfSharedT[timeToRun-1],maxMem/sharedMem);
        }
    }


    free(data);
    free(padded);
    free(paddedT);
    free(filtered);


    cudaFree(d_data);
    cudaFree(d_filtered);
    cudaFree(d_filtered_sh);
    cudaFree(d_filteredT);
    cudaFree(d_filtered_shT);
    cudaFree(d_padded);
    cudaFree(d_paddedT);

}

// applies the algorithm on a file with given variables
void runFile(char const *argv[]){

    GpuTimer timer;

    FILE * png =        fopen(argv[1],"r");
    int N =             atoi(argv[2]);
    int M =             atoi(argv[3]);
    int window =        atoi(argv[4]);
    float patchSigma =  atof(argv[5]);
    float filtSigma =   atof(argv[6]);
    int blockSize =     atoi(argv[7]);


    int sharedMem = (blockSize + 2*blockSize*window*window)*sizeof(float);

    if( png == NULL ){
        printf("Couldn't load png\n");
        exit(1);
    }

    float * data     =  (float*)malloc(N*M*sizeof(float));
    float * filtered =  (float*)malloc(N*M*sizeof(float));
    float * padded   =  (float*)malloc(N*M*window*window*sizeof(float));
    float * paddedT  =  (float*)malloc(N*M*window*window*sizeof(float));

    for( int i=0; i<N*M; i++){
        if(!fscanf(png,"%f,",&data[i])) break;
    }
    fclose(png);


    float *d_data, *d_paddedT, *d_filtered_shT;
    cudaMalloc((void **)&d_data,N*M*sizeof(float));
    cudaMalloc((void **)&d_paddedT,N*M*window*window*sizeof(float));
    cudaMalloc((void **)&d_filtered_shT,N*M*sizeof(float));

    cudaMemcpy(d_data,data,N*M*sizeof(float),cudaMemcpyHostToDevice);

    patchCube(data,padded,N,M,window,patchSigma);

    for(int i=0; i<N*M; i++){
        for(int j=0; j<window*window; j++){
            paddedT[j*N*M +i] = padded[i*window*window + j];
        }
    }

    cudaMemcpy(d_paddedT,paddedT,N*M*window*window*sizeof(float),cudaMemcpyHostToDevice);

    int numOfBlocks = ceil((float)N*M/blockSize);

    timer.Start();
    nlmSharedT<<<numOfBlocks,blockSize,sharedMem>>>(d_filtered_shT,d_data,d_paddedT,N,M,window,filtSigma);
    timer.Stop();

    cudaMemcpy(filtered,d_filtered_shT,N*M*sizeof(float),cudaMemcpyDeviceToHost);

    printf("sharedT: %g ms, dif=%f\n",timer.Elapsed(),maxError(data,filtered,N,M));


    FILE * filtPng =    fopen(argv[8],"w");
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
    free(paddedT);
    free(filtered);

    cudaFree(d_data);
    cudaFree(d_filtered_shT);
    cudaFree(d_paddedT);

}


int main(int argc, char const *argv[]){
    if(argc<9 || argc==10 || argc>11){
        printf("\nPerformance testing usage:\n"
               "11 input arguments (inlcuding executable)\n"
               "./nlm_cuda Seed(0 for random) kernelsToRun(e.g. 1010) N M patch(3/5/..) "
               "patchSigma filtSigma blockSize timesToRun printStats\n"
               "\nFile filtering usage:\n"
               "9 input arguments (inlcuding executable)\n"
               "./nlm_cuda csvToRead N M patch(3/5/..) patchSigma filtSigma blockSize"
               "csvToWrite\n");
        exit(1);
    }

    if(argc==11) randTest(argv);
    if(argc==9)   runFile(argv);
    return 0;
}