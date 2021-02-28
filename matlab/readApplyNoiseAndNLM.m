function [I,J,If,change] = readApplyNoiseAndNLM(fileName,sigma,window,patchSigma,filtSigma,blockSize)
    
    if(mod(window-1,2)~=0)
        disp('Error: Window must be 1+2x (3/5/7/...)')
        return
    end
    
    normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
    
    disp('Reading image...')
    I = imread(convertStringsToChars("../data/"+fileName));
    
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    
    I = normImg(single(I));
   
    [N,M] = size(I);
        
    % crop to divide blockSize exacly
    M = (floor(M/blockSize)*blockSize);

    I = I(:,1:M);  
    
    figure; imagesc(I); title("Original"); colormap gray
    
    noiseParams = {'gaussian',0,sigma};

    disp("Applying noise...")
    J = imnoise( I, noiseParams{:} );
    
    figure; imagesc(J); title("Noisy"); colormap gray
    
    
    disp("Starting NLM...")
    If = nlm_cuda(J,N,M,window,patchSigma,filtSigma,blockSize);
    
    for i=1:length(I(:))
        if(isnan(If(i)))
            If(i)=J(i);
        end
    end
    
    disp("Finished")
    figure; imagesc(If); title("Denoised"); colormap gray
    
    change = If-J;
    
    figure; imagesc(change); title("Difference"); colormap gray; colorbar
    
    RMSE = rms(I(:)-If(:));
    PSNR = psnr(If,I);
    disp(['RMSE: ',num2str(RMSE),' // PSNR: ',num2str(PSNR)])
   
    
    mkdir("./results/"+fileName);
    imwrite(im2uint16(I),convertStringsToChars("./results/"+fileName+"/original.png"));
    imwrite(im2uint16(J),convertStringsToChars("./results/"+fileName+"/noisy.png"));
    imwrite(im2uint16(If),convertStringsToChars("./results/"+fileName+"/denoised.png"));
    imwrite(im2uint16(normImg(change)),convertStringsToChars("./results/"+fileName+"/change.png"));
end
