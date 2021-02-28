function If = nlm_cuda(J,N,M,window,patchSigma,filtSigma,blockSize)

    patchSize = [window window];
    % create 3-D cube with local patches
    patchCube = @(X,w) ...
        permute( ...
            reshape( ...
                im2col( ...
                    padarray( ...
                        X, ...
                        (w-1)./2, 'symmetric'), ...
                    w, 'sliding' ), ...
                [prod(w) size(X)] ), ...
            [2 3 1] );
  
    % create 3D cube
    cube = patchCube(J, patchSize);
    [m, n, d] = size( cube );
    cube = reshape(cube, [ m*n d ] );
  
    % gaussian patch
    H = single(fspecial('gaussian',patchSize, patchSigma));
    H = H(:) ./ max(H(:));
  
    % apply gaussian patch on 3D cube
    cube = bsxfun( @times, cube, H' );
  
    kernel = parallel.gpu.CUDAKernel('../justSharedT.ptx','../justSharedT.cu');
    
    kernel.GridSize = ceil(N*M/blockSize);
    kernel.ThreadBlockSize = blockSize;
    kernel.SharedMemorySize = (blockSize + 2*blockSize*window*window)*4;
    
    If = gpuArray(single(zeros([N M])));
    
    tic;
    If = gather(feval(kernel,If,J,cube,N,M,window,filtSigma));
    toc

end
