function [ cluster ] = mapper_sc( data,filIdx )
%This function implements the MAPPER algorithm. K-mean clustering is
%chosen as the clustering algorithm.
%   Input:
%       - data                  -- the data point [nbPoints * nbDim]
%       - filIdx                -- filtration interval index
% 
%   Output:
%       - cluster               -- matrix [nbPoints * nbInterval]
%

nbInterval = size(filIdx,2);

% initial the output matrix
cluster = zeros(size(filIdx));

% approximately, the number of pixels in each cluster
% pixelsOfClusts = 300;

% number of clusters for each interval
% k = 3*ones(1,size(filIdx,2));


% clustering using k-means. could use other clustering algorithm
for i = 1:nbInterval
    if sum(filIdx(:,i)==1)==0
        continue;
    elseif i==1 && sum(filIdx(:,i)==1)==1
        cluster(filIdx(:,i)==1,i) = 1;
        continue;
    elseif sum(filIdx(:,i)==1)==1
        cluster(filIdx(:,i)==1,i) = 1 + max(cluster(:,i-1));
        continue;
    end
    dataTemp = data(filIdx(:,i)==1,:);
    distMat = squareform(pdist(dataTemp));    
    W = exp( -distMat./std(distMat(:)) );
    W(eye(size(W))~=0) = 0;
    
    [C, ~, ~, ~] = spectralClustering_mapper(W,2);
    
    
    cluster(filIdx(:,i)==1,i) = C;
    
    
    if i > 1
        cluster(:,i) = cluster(:,i) + max(max(cluster(:,1:i-1))).*(cluster(:,i)~=0);
    end
    
end

end

