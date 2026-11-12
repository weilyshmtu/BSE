function [S, F, time_used] = ercan_cpu(X, k, eta, opts)
% ERCAN: Entropy Regularization for unsupervised Clustering with Adaptive Neighbors
% Reference: Wang et al., Pattern Recognition 125 (2022) 108517.
%
% Inputs:
%   X   - Data matrix (n x d), n is the number of samples, d is feature dim
%   c   - Number of clusters
%   k   - Number of adaptive neighbors (l0-norm constraint)
%   eta - Ratio coefficient for initial gamma (typically 0.1 to 10)

tic;
n = opts.n;
c = opts.n_clusters;

% --- Step 1: Compute pairwise squared Euclidean distance ---
Dx = opts.Dx;

% --- Step 2: Initialize gamma, lambda and S ---
% gamma = (eta / n) * sqrt(sum(d_ij^2))
gamma = (eta / n) * sqrt(sum(Dx(:).^2));
lambda = gamma;

% Initialize S using Eq. (21)
S = zeros(n, n);
for i = 1:n
    d_i = Dx(i, :);
    d_i(i) = inf; % exclude self-loop
    s_i = exp(-d_i / gamma);
    S(i, :) = s_i / sum(s_i);
end
S_sym = (S + S') / 2;
D_s = diag(sum(S_sym, 2));
L_s = D_s - S_sym;
[F, ~, ev] = eig1(L_s, c, 0); 

maxIter = 100;
tol = 1e-10;

% --- Main Alternative Optimization Loop ---
for iter = 1:maxIter
    % 1. Update F
    S_sym = (S + S') / 2;
    D_s = diag(sum(S_sym, 2));
    L_s = D_s - S_sym;
    
    
    F_old = F;
    [F, ~, ev] = eig1(L_s, c, 0); 
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1)); 

    flag = sum(logical(diag(D_s-eye(n))<0.01)); % check  whether the graph is doubly stochastic
    if fn1 > tol 
        eta = 2*eta;
        gamma = (eta / n) * fro_norm_Dx;
        lambda = gamma;
    elseif fn2 < tol
        eta = eta/2; 
        gamma = (eta / n) * fro_norm_Dx;
        lambda = gamma;
        F = F_old;
    elseif fn1 < tol && fn2 > tol && flag == n  % output graph in advance [when the graph is structured and doubly stochastic] 
        time_used = toc;
        break;
    end
    
    % 2. Update S (with l0-norm constraint and entropy regularization)
    sumF = sum(F(:, 1:c).^2, 2);
    Df = bsxfun(@plus, sumF, sumF') - 2 * (F(:, 1:c) * F(:, 1:c)');
    Df(Df < 0) = 0;
    
    D_all = Dx + lambda * Df;
    S = zeros(n, n);
    
    % Eq. (35): Monotonic function optimization for S
    for i = 1:n
        d_i = D_all(i, :);
        d_i(i) = inf; % Ignore self-distance
        
        % Extract k nearest neighbors
        [sorted_d, idx] = sort(d_i, 'ascend');
        idx_k = idx(1:k);
        dk = sorted_d(1:k);
        
        % Compute similarity with entropy regularization
        s_val = exp(-dk / gamma);
        s_val = s_val / sum(s_val);
        
        S(i, idx_k) = s_val;
    end
end

if iter >= maxIter
    time_used = toc;
end
end

