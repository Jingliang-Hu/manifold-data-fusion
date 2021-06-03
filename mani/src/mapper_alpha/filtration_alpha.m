function [ filIdx ] = filtration_alpha( filVal,overlap,nbInterval,flag )
%This function calculate the equal sized filtration intervals for the
%MAPPER algorithm
%   Input:
%       - filVal                     -- filtration values
%       - overlap                    -- overlap rate of adjacent interval
%       - nbInterval                 -- number of intervals
%       - flag                       -- equal interval(1); statistical interval(2)
%
%   Output:
%       - filIdx                     -- index of points of each interval
%

[nb,dn,~] = size(filVal);
filIdx = zeros(nb,nbInterval^dn);


if dn == 1
    [ filIdx ] = oneDFiltration( filVal,overlap,nbInterval,flag );
elseif dn == 2
    [ filIdx ] = twoDFiltration( filVal(:,1),filVal(:,1),overlap,nbInterval,nbInterval,flag );
else
    disp('Only 1d and 2d filtration is supported by far');
end

end

