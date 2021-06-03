X = x_pc;
filVal = x_pc(:,1);
nbBins = 10;
oRate = 0.5;

city = [1;cumsum(nbOfSamples)];


for i = 1:10
    [ visualW,W ] = mapper_alpha( X(city(i):city(i+1),:),nbBins,oRate,filVal(city(i):city(i+1)),y_tra(city(i):city(i+1))'+1 );
end