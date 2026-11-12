function S = initialize_S(X, D, k, issymmetric)
% construct similarity matrix with probabilistic k-nearest neighbors. 
% It is a parameter free, distance consistent similarity.
% ----------------------------------------------------------------
% X: each column is a data point
% k: number of neighbors
% S: similarity matrix
% issymmetric: set S = (S+S')/2 if issymmetric=1
% Ref: F. Nie, X. Wang, M. I. Jordan, and H. Huang, The constrained
% Laplacian rank algorithm for graph-based clustering, in AAAI, 2016.

if nargin < 4
    issymmetric = 1;
end
if nargin < 3
    k = 5;
end

[~, n] = size(X);
% D = L2_distance_1(X, X);
[~, idx] = sort(D, 2); % sort each row

S = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

if issymmetric == 1
    S = (S+S')/2;
end







