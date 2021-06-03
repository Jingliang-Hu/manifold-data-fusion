function [ clusterW ] = simpleVisualW( mapperCluster )
%This function calculate the connecting matrix W of the simplified graph
%   Input:
%       - mapperCluster         -- clusters derived by mapper
%
%   Output:
%       - clusterW              -- a c by c connecting matrix, c: number of clusters


clusterW = zeros(max(mapperCluster(:)),max(mapperCluster(:)));
for i = 1:size(mapperCluster,1)
    temp = unique(mapperCluster(i,:));
    temp(temp==0) = [];
        if length(temp)>1
            C = nchoosek(temp,2);
            for j = 1:size(C,1)
                clusterW(C(j,1),C(j,2)) = 1;
                clusterW(C(j,2),C(j,1)) = 1;
            end
        end
end

end

