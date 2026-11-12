function [S, time_used] = sgl_gpu(A, k, lambda)
% SGL: Structured Graph Learning
% GPU Version: Optimized for parallel simplex projection and eigenvalue decomposition.

tic;
n = size(A, 1);
% Cast to double precision for higher numerical stability during eigenvalue decomposition
A = gpuArray(double(full(A))); 

% Initialize S as the symmetric version of A on GPU
S = (A + A') / 2;

maxIter = 50;
tol = 1e-6;

% Pre-generate a tiny shift matrix to ensure the Laplacian is non-singular
% This prevents eigs from failing on the zero-eigenvalue inherent to Laplacian matrices
shift_matrix = gpuArray(1e-5 * eye(n, 'double'));

for iter = 1:maxIter
    % --- Update F (Eigenvectors of Laplacian) ---
    L = -S;
    L(1:n+1:end) = L(1:n+1:end) + sum(S, 2)';
    L = (L + L') / 2; % Ensure perfect symmetry to avoid complex eigenvalues
    
    % Apply shift to the Laplacian matrix to ensure stability
    L_shifted = L + shift_matrix;
    
    try
        % Perform eigenvalue decomposition on GPU
        [V_sub, D_sub] = eigs(L_shifted, k, 'smallestabs', struct('disp', 0));
    catch
        % Fallback mechanism if GPU eigs fails
        warning('GPU eigs failed at iteration %d, falling back to CPU.', iter);
        [V_sub, D_sub] = eigs(gather(L_shifted), k, 'smallestabs', struct('disp', 0));
        V_sub = gpuArray(V_sub);
    end
    
    % Extract the eigenvectors corresponding to the k smallest eigenvalues
    [~, idx] = sort(diag(D_sub), 'ascend');
    F = V_sub(:, idx);
    
    % --- Update S (Vectorized Simplex Projection) ---
    S_old = S;
    
    % Precompute distance matrix in F space on GPU
    distF = L2_distance_gpu(F', F');
    
    % V calculation: incorporate the structural learning term
    V = A - (lambda / 4) * distF;
    
    % Vectorized row-wise projection onto the simplex
    S = project_to_simplex_matrix_gpu(V);
    
    % Maintain symmetry for the graph weight matrix
    S = (S + S') / 2; 
    
    % Check for convergence
    if norm(S - S_old, 'fro') / norm(S_old, 'fro') < tol
        break;
    end
end

time_used = toc;
% Gather final result back to CPU memory
S = gather(S);
end

%% --- GPU Optimized Helper Functions ---
function D = L2_distance_gpu(a, b)
    % Efficient L2 distance calculation using matrix multiplication
    aa = sum(a.*a, 1); 
    bb = sum(b.*b, 1);
    ab = a' * b;
    D = max(0, aa' + bb - 2 * ab);
end

function X_proj = project_to_simplex_matrix_gpu(X)
    % Efficient row-wise projection onto the probability simplex
    [n, d] = size(X);
    Xs = sort(X, 2, 'descend');
    Xsum = cumsum(Xs, 2);
    
    idx = gpuArray(double(1:d)); 
    cond = Xs + (1 - Xsum) ./ idx > 0;
    rho_idx = sum(cond, 2);
    
    lin_idx = (double(rho_idx) - 1) * double(n) + double((1:n)');
    Xsum_T = Xsum';
    theta = (1 - Xsum_T(lin_idx)) ./ double(rho_idx);
    
    X_proj = max(X + theta, 0);
end