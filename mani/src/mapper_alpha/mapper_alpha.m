function [ visualW,W,clusterW ] = mapper_alpha( X,nbBins,oRate,filVal,colorIdx )
%This function derives topological structure of data X by mapper algorithm
%   Input:
%       - X             -- data n*m, n samples and m features
%       - nbBins        -- number of bins
%       - oRate         -- overlapping rate
%       - filVal        -- filtration value n*d, n samples and d filters
%
%   Output:
%       - clusterW      -- a c by c connecting matrix, c: number of clusters
%       - W             -- connecting matrix

visualW = 0;
W = 0;

% get filtration interval
flag=2;
[ filIdx ] = filtration_alpha( filVal,oRate,nbBins,flag );

% clustering
[ mapperCluster ] = mapper_sc( X,filIdx );

% simple visualization matrix
% [ visualW ] = simpleVisualW( mapperCluster );

% plot simplified graph
[ clusterW ] = simpleVisual_label( mapperCluster,colorIdx,X(:,1:2) );

% connecting matrix of all data
% [ W ] = mapperW( mapperCluster );

end

