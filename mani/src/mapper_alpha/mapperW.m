function [ W ] = mapperW( mapperCluster )
%This function calculates the connecting matrix of all data points
%   Input:
%       - mapperCluster         -- clusters derived by mapper
%
%   Output:
%       - W              -- a c by c connecting matrix, c: number of data points

W = zeros(size(mapperCluster,1),size(mapperCluster,1));
for i = 1:size(mapperCluster,2)
    temp = (repmat(mapperCluster(:,i),1,length(mapperCluster(:,i)))==repmat(mapperCluster(:,i)',length(mapperCluster(:,i)),1));
    temp(mapperCluster(:,i)==0,:)=0;
    W = W + temp;
end
W = W>0;
W(eye(size(W))==1) = 0;


end

