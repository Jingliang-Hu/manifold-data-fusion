function [ clusterW ] = simpleVisual_label( mapperCluster,colorIdx, axisData )
%This function calculate the connecting matrix W of the simplified graph
%   Input:
%       - mapperCluster         -- clusters derived by mapper
%
%   Output:
%       - clusterW              -- a c by c connecting matrix, c: number of clusters

map=[
% % R     G       B                      COLOR       NAME        DESCRIPTION         MINIMUM             MAXIMUM
165,   0,       33;               % % 140         "1"         "compHR"            1.000000            1.000000
204,   0,        0;               % % 209         "2"         "compMR"            2.000000            2.000000
255,   0,        0;               % % 255         "3"         "compLR"            3.000000            3.000000
153,   51,       0;               % % 19903       "4"         "openHR"            4.000000            4.000000
204,   102,      0;               % % 26367       "5"         "openMR"            5.000000            5.000000
255,   153,      0;               % % 5609983     "6"         "openLR"            6.000000            6.000000
255,   255,      0;               % % 388858      "7"         "light"             7.000000            7.000000
192,   192,    192;               % % 12369084    "8"         "largeLow"          8.000000            8.000000
255,   204,    153;               % % 11193599    "9"         "sparse"            9.000000            9.000000
 77,    77,     77;               % % 5592405     "10"        "industr"           10.000000           10.000000
  0,   102,      0;               % % 27136       "A"         "denseTree"         101.000000          101.000000
 21,   255,     21;               % % 43520       "B"         "scatTree"          102.000000          102.000000
102,   153,      0;               % % 2458980     "C"         "bush"              103.000000          103.000000
204,   255,    102;               % % 7986105     "D"         "lowPlant"          104.000000          104.000000
  0,     0,    102;               % % 0           "E"         "paved"             105.000000          105.000000
255,   255,    204;               % % 11466747    "F"         "soil"              106.000000          106.000000
 51,   102,    255;               % % 16738922    "G"         "water"             107.000000          107.000000
];



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
        clusterL(cluIdx(j),1) = mean(axisData(mapperCluster(:,i)==cluIdx(j),1));
        clusterL(cluIdx(j),2) = mean(axisData(mapperCluster(:,i)==cluIdx(j),2));        
    end
end




clusterCol = zeros(max(mapperCluster(:)),3);
for i = 1:size(clusterCol,1)
    index = sum(mapperCluster==i,2);
    clusterCol(i,:) = map(mode(colorIdx(index==1)),:);
    disp(i)
end
[IND,map] = rgb2ind(permute(uint8(clusterCol),[1,3,2]),32);
figure,p = plot(graph(clusterW),'XData',clusterL(:,1),'YData',clusterL(:,2),'NodeCData',IND);
colormap(map);
p.MarkerSize = 5;


end

