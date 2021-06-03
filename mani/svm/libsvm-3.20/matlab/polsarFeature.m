function [ feature ] = polsarFeature( polsar )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

[r,c,d]=size(polsar);
feature=zeros(r,c,d);
feature(:,:,1:2)=polsar(:,:,1:2);
feature(:,:,3)=abs(polsar(:,:,3)+1i*polsar(:,:,4));
feature(:,:,4)=angle(polsar(:,:,3)+1i*polsar(:,:,4));


end

