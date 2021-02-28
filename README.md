# Non-local Means Acceleration With Cuda

### Performance testing usage:
11 input arguments (inlcuding executable)
./nlm_cuda Seed(0 for random) kernelsToRun(e.g. 1010) N M patch(3/5/..) patchSigma filtSigma blockSize timesToRun printStats

kernelsToRun: Specifies which kernels to run in binary, numbered from 1 to 4. 1 to run 0 to not.  
  1: Using Global memory  
  2: Using Global memory coalesced  
  3: Using Shared memory  
  4: Using Shared memory coalesced  
  
N/M: Array dimensions.  
patch: Patch window side size, must be odd.  
patchSigma: Variable for the Gauss patch intensity.  
filtSigma: Variable for the filter's intensity.  
blockSize: How to break up pixels into thread blocks. Array x dimension MUST be multiple of this for shared algoriths (3-4)  
timesToRun: How many times to run the kernels. Usefull for stats and averaging.  
printStats: If 1, prints some usefull stats from the time measurements.  


### File filtering usage:
9 input arguments (inlcuding executable)  
./nlm_cuda csvToRead N M patch(3/5/..) patchSigma filtSigma blockSize csvToWrite  

csvToRead:  Path to the input csv array  
csvToWrite: Path to the output csv array  

### For usage with actual images
Use the **readApplyNoiseAndNLM** matlab script.  
