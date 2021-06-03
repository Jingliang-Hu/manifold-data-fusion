function eigvector= GGF(X, Xsp, Xli,no_dims, k)

%
%   [eigvector, GG]= GGF(X, Xsp, Xli, Xemp, no_dims, k)
%
% Generalized Graph-Based Fusion of Hyperspectral and LiDAR Data Using Morphological Features
%
%   W. Liao, R. Bellens, A. Pizurica, S. Gautama, W. Philips, â€œGeneralized Graph-Based Fusion of 
%   Hyperspectral and LiDAR Data Using Morphological Features,â€? IEEE Geoscience and Remote Sensing 
%   Letters, vol. 12, no. 3, pp. 552-556, Mar. 2015.
%
% %       Copyright notes
% %       Author: Wenzhi Liao, IPI, Telin, Ghent University, Belgium
% %       Date: 10/1/2011
  

    % Construct neighborhood graph
        G = L2_distance_GGF(X', X');
        Gsp = L2_distance_GGF(Xsp', Xsp');
        Gli = L2_distance_GGF(Xli', Xli');
        [~, indsp] = sort(Gsp);
        [~, indli] = sort(Gli); 
        k1=150;
        for i=1:size(G, 1)
            Gsp(i, indsp((2 + k1):end, i)) = 0;
            Gli(i, indli((2 + k1):end, i)) = 0; 
            
            Gsp(i, indsp(2:(k1+1), i)) = 1;
            Gli(i, indli(2:(k1+1), i)) = 1; 
        end

        FuG=Gsp.*Gli;
        FuG=~FuG;
        G=G+FuG*max(G(:));
        [tmp, ind] = sort(G);
        for i=1:size(G, 1)
           G(i, ind((2 + k):end, i)) = 0;
        end

    G = max(G, G');
    
    % ------------------------------------------
    % modified by Jingliang
    % Compute Gaussian kernel (heat kernel-based weights)
    G(G ~= 0) = exp(-G(G ~= 0));
    % ------------------------------------------
    
    
    D = diag(sum(G, 2));
    
    % Compute Laplacian
    L = D - G;
    L(isnan(L)) = 0; D(isnan(D)) = 0;
	L(isinf(L)) = 0; D(isinf(D)) = 0;

    % Compute XDX and XLX and make sure these are symmetric
    DP = X' * D * X;
    LP = X' * L * X;
    DP = (DP + DP') / 2;
    LP = (LP + LP') / 2;

    % Perform eigenanalysis of generalized eigenproblem (as in LEM)
   [eigvector, eigvalue] = eig(LP, DP);

    
    % Sort eigenvalues in descending order and get largest eigenvectors
    [eigvalue, ind] = sort(diag(eigvalue), 'ascend');
    eigvector = eigvector(:,ind(1:no_dims));
        
