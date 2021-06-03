function [ W ] = mapper_ssma_wt( data, varargin )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
%   Inputs:  data, nbInterval, overlap, filVal1, filVal2, nbfilt, flag

if nargin > 1
    
end



if nargin == 1
    % pca decomposition
    [T,~,~,~,~,~]=pca(data);
    filVal1 = data*T(:,1);
    filVal2 = data*T(:,2);
    overlap = 0.5;
    nbInterval1 = 5;
    nbInterval2 = 5;
    flag = 2;
end



% % calculating filtration interval
% [ filIdx ] = oneDFiltration( filVal1,overlap,nbInterval1,flag );

[ filIdx ] = twoDFiltration( filVal1,filVal2,overlap,nbInterval1,nbInterval2,flag );


% mapper
[ mapperCluster ] = mapper_sc( data,filIdx );


W = zeros(size(mapperCluster,1),size(mapperCluster,1));
for i = 1:size(mapperCluster,2)
    temp = (repmat(mapperCluster(:,i),1,length(mapperCluster(:,i)))==repmat(mapperCluster(:,i)',length(mapperCluster(:,i)),1));
    temp(mapperCluster(:,i)==0,:)=0;
    W = W + temp;
end
W = W>0;
W(eye(size(W))==1) = 0;
figure,plot(graph(W),'XData',filVal1,'YData',filVal2)

end

