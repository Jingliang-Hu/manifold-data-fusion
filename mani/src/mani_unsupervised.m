function [ map1,map2 ] = mani_unsupervised( Wp,Z,dn1,dn2 )
%UNTITLED2 Summlabry of this function goes here
%   Detlabiled expllabnlabtion goes here
epsilon = 1e-5;

[ Wpn ] = WNormalize( Wp );

I = eye(length(sum(Wp)));

Lp = I - Wpn;

d = dn1+dn2;

[u, s, ~]=svd(full(Z*Z'));
F=u*sqrt(s);
Fplus=pinv(F);

clear u s;
T=Fplus*Z*Lp*Z'*Fplus'; 
% T=Z*(mu*Lp)*Z';
% T=Z*(Ls+mu*Lp)*Z'; 

%~~~eigen decomposition~~~
T=0.5*(T+T');
[ev, ea]=eigs(full(T),d);
clear T Z F;

%sorting ea by ascending order
ea=diag(ea);
[~, index]  =sort(ea);
ea =ea(index); ev=ev(:,index);
ev =Fplus'*ev;
ev = ev./repmat(sqrt(sum(ev.^2, 1)),size(ev,1),1);

idx = ea>epsilon;
map1=ev(1:dn1,idx);
map2=ev(dn1+1:dn1+dn2,idx);

end

