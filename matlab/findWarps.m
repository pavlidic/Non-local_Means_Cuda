x = [3,5,7,9];
y = [8,16,32,64,128];

for i=1:4
    for j=1:5
        w(i,j)=warps(y(j),x(i));
    end
end



function out = warps(blockSize,window)
    maxB = 49152;
    shared = (blockSize + 2*blockSize*window*window)*4;
    out = floor(maxB/shared);
    
    
    
    
end