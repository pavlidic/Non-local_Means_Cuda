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
