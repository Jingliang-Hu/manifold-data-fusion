function [ clu ] = mapper_sl( data,filIdx )
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
clu = zeros(size(filIdx));

KK = 50;
% clustering using the one in original paper
for i = 1:nbInterval
    if sum(filIdx(:,i)==1)==0
        continue;
    elseif i==1 && sum(filIdx(:,i)==1)==1
        clu(filIdx(:,i)==1,i) = 1;
        continue;
    elseif sum(filIdx(:,i)==1)==1
        clu(filIdx(:,i)==1,i) = 1 + max(clu(:,i-1));
        continue;
    end
    dataTemp = data(filIdx(:,i)==1,:);
    Z = linkage(dataTemp);
    [x,y] = hist(Z(:,3),KK);
    ndx = find(x==0);
    if isempty(ndx)
        disp("KK has to be adjusted, clustering is not accomplished")
        break;
    end
    C = cluster(Z,'cutoff',y(ndx(1)));   
    
    clu(filIdx(:,i)==1,i) = C;
    
    
    if i > 1
        clu(:,i) = clu(:,i) + max(max(clu(:,1:i-1))).*(clu(:,i)~=0);
    end
    
end

end


