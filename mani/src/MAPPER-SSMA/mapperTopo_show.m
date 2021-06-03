function [ T ] = mapperTopo_show( data )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% pca decomposition
[T,s,~,~,pbl,~]=pca(data);
filVal1 = data*T(:,1);
filVal2 = data*T(:,2);


overlap = 0.5;
nbInterval1 = 5;
nbInterval2 = 5;
flag = 2;


red4x = linspace(0,1,length(filVal1));
blue4y = linspace(0,1,length(filVal2));

[~,xloc] = sort(filVal1);
[~,yloc] = sort(filVal2);

xcol = zeros(size(filVal1));
ycol = zeros(size(filVal2));

xcol(xloc) = red4x;
ycol(yloc) = blue4y;

figure,scatter(filVal1,filVal2,[],(xcol+ycol)./2,'filled')




[ filIdx ] = twoDFiltration( filVal1,filVal2,overlap,nbInterval1,nbInterval2,flag );

figure,
id = 1;
for j = 1:nbInterval1%:-1:1
    for i = nbInterval2:-1:1  
        subplot(nbInterval1,nbInterval2,(i-1)*nbInterval1+j);
        scatter(filVal1(filIdx(:,id)>0),filVal2(filIdx(:,id)>0),[],(xcol(filIdx(:,id)>0)+ycol(filIdx(:,id)>0))./2,'filled');
        id = id + 1;
        set(gca,'xtick',[])
        set(gca,'ytick',[])
    end
end


% mapper
[ mapperCluster ] = mapper_sc( data,filIdx );


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


clusterL = zeros(max(mapperCluster(:)),3);
for i = 1:size(mapperCluster,2)    
    cluIdx = unique(mapperCluster(:,i));
    cluIdx(cluIdx==0) = [];
    for j = 1:length(cluIdx)
        clusterL(cluIdx(j),1) = mean(filVal1(mapperCluster(:,i)==cluIdx(j)));
        clusterL(cluIdx(j),2) = mean(filVal2(mapperCluster(:,i)==cluIdx(j)));        
    end
end

red4x = linspace(0,1,length(clusterL(:,1)));
blue4y = linspace(0,1,length(clusterL(:,2)));

[~,xloc] = sort(clusterL(:,1));
[~,yloc] = sort(clusterL(:,2));

xcol = zeros(size(clusterL(:,1)));
ycol = zeros(size(clusterL(:,2)));

xcol(xloc) = red4x;
ycol(yloc) = blue4y;


clusterW(eye(size(clusterW))==1) = 0;
figure,p = plot(graph(clusterW),'XData',clusterL(:,1),'YData',clusterL(:,2),'NodeCData',(xcol+ycol)./2);
p.MarkerSize = 5;
figure,plot(graph(clusterW))








T = zeros(size(mapperCluster,1),size(mapperCluster,1));
for i = 1:size(mapperCluster,2)
    temp = (repmat(mapperCluster(:,i),1,length(mapperCluster(:,i)))==repmat(mapperCluster(:,i)',length(mapperCluster(:,i)),1));
    temp(mapperCluster(:,i)==0,:)=0;
    T = T + temp;
end
T = T>0;
T(eye(size(T))==1) = 0;
figure,plot(graph(T),'XData',filVal1,'YData',filVal2)

end

