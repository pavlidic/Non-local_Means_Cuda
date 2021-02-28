If = csvread("../data/houseclean.csv");
c  = csvread("../data/filtered.csv");

max(max(abs(c-If)))

figure
imagesc(c-If)
colormap gray
colorbar
title('Cuda - Matlab difference')

figure
imagesc(c)
colormap gray
title('Cuda NLM')

figure
imagesc(If)
colormap gray
title('Matlab NLM')