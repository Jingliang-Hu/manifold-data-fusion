function [ T ] = mapperTopo_parameter_sl( data,bin )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% pca decomposition

% data = data./repmat(sum(data,2),1,size(data,2));
% data = zscore(data);
% [T,s,~,~,pbl,~]=pca(data);
% filVal1 = data*T(:,1);
% filVal2 = data*T(:,2);

K = 10;
D = pdist(data);
D = squareform(D);
D = sort(D);
filVal1 = 1./D(K+1,:)';



overlap = 0.5;
nbInterval1 = bin;
nbInterval2 = bin;
flag = 2;

% % calculating filtration interval
[ filIdx ] = oneDFiltration( filVal1,overlap,nbInterval1,flag );

% [ filIdx ] = twoDFiltration( filVal1,filVal2,overlap,nbInterval1,nbInterval2,flag );


% mapper
[ mapperCluster ] = mapper_sl( data,filIdx );


T = zeros(size(mapperCluster,1),size(mapperCluster,1));
for i = 1:size(mapperCluster,2)
    temp = (repmat(mapperCluster(:,i),1,length(mapperCluster(:,i)))==repmat(mapperCluster(:,i)',length(mapperCluster(:,i)),1));
    temp(mapperCluster(:,i)==0,:)=0;
    T = T + temp;
end
T = T>0;
T(eye(size(T))==1) = 0;
% figure,plot(graph(T),'XData',filVal1,'YData',filVal2)

end


