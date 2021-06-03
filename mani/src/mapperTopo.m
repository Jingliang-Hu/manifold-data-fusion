function [ T ] = mapperTopo( data )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% pca decomposition
[T,~,~,~,~,~]=pca(data);
filVal1 = data*T(:,1);
filVal2 = data*T(:,2);


overlap = 0.5;
nbInterval1 = 5;
nbInterval2 = 5;
flag = 2;

% % calculating filtration interval
% [ filIdx ] = oneDFiltration( filVal1,overlap,nbInterval1,flag );

[ filIdx ] = twoDFiltration( filVal1,filVal2,overlap,nbInterval1,nbInterval2,flag );


% mapper
[ mapperCluster ] = mapper_sc( data,filIdx );


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

