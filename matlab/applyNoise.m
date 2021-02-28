function J = applyNoise( I, sigma)
    
    noiseParams = {'gaussian', ...
                 0,...
                 sigma};

    
    J = imnoise( I, noiseParams{:} );
    
    figure
    imagesc(J)
    colormap gray
    
    dlmwrite("../data/"+fileName+"_dirty.csv", J, 'delimiter', ',', 'precision', 15);

end
