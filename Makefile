SHELL := /bin/bash

NVCC=nvcc
GCC=gcc

CFLAGS = #-arch=sm_61#-gencode arch=compute_61,code=sm_61 -lineinfo -use_fast_math #-Xptxas -O3,-v#-O3 #-Xptxas -dlcm=ca

all: nlm_cuda just_sharedT


nlm_cuda: nlm_cuda.cu extra.cpp
	$(NVCC) $(CFLAGS) -o $@ $^ 


just_sharedT: justSharedT.cu
	$(NVCC) $(CFLAGS) -ptx $^


test_perf:
	./nlm_cuda 0 1111 256 256 5 1 0.02 32 5 1


test_house:
	./nlm_cuda ./data/housedirty.csv 64 64 3 1.66666666 0.02 32 ./data/filtered.csv


matlab_test:
	matlab -nodisplay -nosplash -nodesktop -r "run('./matlab/pipeline_non_local_means.m');exit;" | tail -n +11
	./nlm_cuda ./data/housedirty.csv 64 64 3 1.66666666 0.02 32 ./data/filtered.csv
	matlab -nodisplay -nosplash -nodesktop -r "run('./matlab/testIfSame.m');exit;" | tail -n +11
