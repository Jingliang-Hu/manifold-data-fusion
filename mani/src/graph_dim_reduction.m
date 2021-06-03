function [ map1 ] = graph_dim_reduction(W,Z)
%UNTITLED2 Summlabry of this function goes here
%   Detlabiled expllabnlabtion goes here
epsilon = 1e-5;

W = max(W,W');
D = diag(sum(W,2));

L = D - W;

% Compute XDX and XLX and make sure these are symmetric
DP = Z' * D * Z;
LP = Z' * L * Z;
DP = (DP + DP') / 2;
LP = (LP + LP') / 2;

% Perform eigenanalysis of generalized eigenproblem (as in LEM)
[ev, ea] = eig(LP, DP);
ev = real(ev);
ea = real(ea);
%sorting ea by ascending order
ea=diag(ea);
[~, index]  =sort(ea);
ea =ea(index); ev=ev(:,index);


idx = ea>epsilon;
map1=ev(:,idx);


end

