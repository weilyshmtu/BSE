function [S, time_used] = sgl_cpu(A, k, lambda)
% A: Initial affinity matrix (n x n)
% k: Number of clusters
% lambda: Regularization parameter for rank constraint
tic;
n = size(A, 1);
% Initialize S as the symmetric version of A
S = (A + A') / 2;
maxIter = 50;
tol = 1e-6;

for iter = 1:maxIter
    % --- Update F (Eigenvectors of Laplacian) ---
    D = diag(sum(S, 2));
    L = D - S;
    % Get the k smallest eigenvectors
    [F, ~, ~] = eigs(L, k, 'smallestreal');
    
    % --- Update S (Row-wise Simplex Projection) ---
    S_old = S;
    % Precompute distance matrix in F space: d_ij = ||f_i - f_j||^2
    distF = L2_distance_1(F', F');
    
    % V = A - (lambda/4)*distF
    V = A - (lambda / 4) * distF;
    
    for i = 1:n
        % Solve: min ||s_i - v_i||^2 s.t. s_i'1=1, s_i>=0
        S(i, :) = EProjSimplex_new(V(i, :));
    end
    S = (S + S') / 2; % Keep symmetry
    
    % Check convergence
    if norm(S - S_old, 'fro') / norm(S_old, 'fro') < tol
        time_used = toc;
        break;
    end
end

if iter >= maxIter
    time_used = toc;
end
end