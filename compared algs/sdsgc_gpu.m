function [C, F, time_used] = sdsgc_gpu(X, k, opts)
% SDSGC: Scalable Doubly Stochastic Graph Clustering (GPU Version)
% HIGH-STABILITY VERSION: Fixed eigenvalue shift bug & prevented numerical collapse.

tic;
maxIter = 300;
n = opts.n;
c = opts.n_clusters;

% --- Input Data Precision Setup ---
Dx = gpuArray(double(full(opts.Dx)));
Dxt = gather(Dx); 
[~, idx] = sort(Dxt, 2); 
C = gpuArray(double(full(opts.S_init)));
d_vec = sum(C, 2);
L = -C; 
L(1:n+1:end) = L(1:n+1:end) + d_vec'; 
[F, ~, ~] = eig1_gpu(L, c, 0); 

tol = 1e-10;
rho = 1.3;
mu = 1;
eta = 10;
local = 1;

LAMBDA = gpuArray.zeros(n, n, 'double'); 
half_Dx = 0.5 * Dx; 

% --- Safe Indexing using int64 for large scale data ---
row_idx = gpuArray(int64(1:n)');
idx_k2 = (repmat(int64(k+2), n, 1) - 1) * int64(n) + row_idx; 
col_indices = gpuArray(int64(idx(:, 2:k+1))); 
row_indices = repmat(row_idx, 1, k);
lin_idx_Dx = (col_indices - 1) * int64(n) + row_indices;
sum_Dx_idx = sum(Dx(lin_idx_Dx), 2);
gamma = sum(k/2 * Dx(idx_k2) - 1/2 * sum_Dx_idx) / double(n);

if local == 1
    mask = gpuArray.false(n, n);
    col_indices_mask = gpuArray(int64(idx(:, 2:k))); 
    row_indices_mask = repmat(row_idx, 1, k-1);
    lin_idx_mask = (col_indices_mask - 1) * int64(n) + row_indices_mask;
    mask(lin_idx_mask) = true;
end

for iter = 1:maxIter
    distF = L2_distance_self_gpu(F'); 
    
    Z = mu*C' - LAMBDA - half_Dx - (0.5*eta)*distF;
    
    % --- Main Loop Numerical Safety Check ---
    if any(isnan(Z), 'all') || any(isinf(Z), 'all')
        warning('SDSGC: Iteration %d detected extreme numerical values (NaN/Inf), triggering early termination.', iter);
        break; 
    end
    
    if local == 1
        Z(~mask) = 0; 
    end
    Z = Z / (gamma + mu);
    
    C = project_to_simplex_matrix_gpu(Z, 2); 
    
    h = C - C';
    LAMBDA = LAMBDA + mu * h;
    
    mu = min(mu * rho, 1e10); 
    
    C_sym = (C + C') / 2;
    d_vec = sum(C_sym, 2);
    
    L_sym = -C_sym; 
    L_sym(1:n+1:end) = L_sym(1:n+1:end) + d_vec';
    
    F_old = F;
    
    % Update Eigenvectors
    [F, ~, ev] = eig1_gpu(L_sym, c, 0);
    
    % [Robustness Fix] Filter out negative noise from GPU floating-point precision
    ev = max(0, ev); 
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1)); 
    
    flag = sum(abs(d_vec - 1) < 0.01); 
    
    % Now that fn1 reflects accurate spectral properties, eta update logic functions correctly
    if fn1 > tol 
        eta = 2 * eta;
    elseif fn2 < tol
        eta = eta / 2;  
        F = F_old;
    elseif fn1 < tol && fn2 > tol && flag == n  
        break;
    end
end
time_used = toc;
C = gather(C); 
F = gather(F);
end

%% --- Helper Functions ---
function [V, D, evs] = eig1_gpu(L, c, is_smallest)
    % Robust eigenvalue solver with numerical correction
    L = (L + L') / 2; 
    n = size(L, 1);
    k_eigs = min(c + 1, n); 
    
    opts_eigs.disp = 0;
    opts_eigs.maxit = 1500; 
    opts_eigs.tol = 1e-5; % Slightly relaxed tolerance for performance
    
    if is_smallest == 0
        sigma = -1e-4; 
        sort_dir = 'ascend';
        shift_val = 1e-4; % Uniform shift value
        L(1:n+1:end) = L(1:n+1:end) + shift_val; 
    else
        sigma = 'largestabs';
        sort_dir = 'descend';
        shift_val = 0;
    end
    
    try
        [V_all, D_all] = eigs(L, k_eigs, sigma, opts_eigs);
        if any(isnan(D_all), 'all')
            error('eigs returned NaN, triggering fallback.');
        end
    catch
        L_cpu = gather(L);
        try
            [V_all_cpu, D_all_cpu] = eigs(double(L_cpu), k_eigs, sigma, opts_eigs);
            if any(isnan(D_all_cpu), 'all')
                error('CPU eigs also returned NaN.');
            end
            V_all = gpuArray(V_all_cpu);
            D_all = gpuArray(D_all_cpu);
        catch
            warning('Sparse eigs failed; falling back to robust dense eig solver.');
            [V_all_cpu, D_all_cpu] = eig(double(gather(L)), 'vector');
            [~, sort_idx] = sort(D_all_cpu, sort_dir);
            
            V_all = gpuArray(V_all_cpu(:, sort_idx(1:k_eigs)));
            D_all = gpuArray(diag(D_all_cpu(sort_idx(1:k_eigs))));
        end
    end
    
    evs_all = diag(D_all);
    
    % [Critical Fix] If solving for smallest eigenvalues, must reverse the artificial shift
    if is_smallest == 0
        evs_all = evs_all - shift_val;
        evs_all(abs(evs_all) < 1e-8) = 0; % Force tiny noise to zero to prevent spectral ranking issues
    end
    
    [evs, i] = sort(evs_all, sort_dir);
    
    V = V_all(:, i(1:c));
    D = evs(1:c);
end

function D = L2_distance_self_gpu(a)
    % Efficient computation of self-L2 distances on GPU
    aa = sum(a.*a, 1); 
    ab = a' * a;
    D = max(0, aa' + aa - 2 * ab);
end

function X_proj = project_to_simplex_matrix_gpu(X, dim)
    % Efficient row/column-wise projection of a matrix onto the probability simplex
    if dim == 1, X = X'; end
    [rows, cols] = size(X); 
    X_proj = gpuArray.zeros(rows, cols, 'double');
    
    block_size = 3000;
    idx_vec = gpuArray(double(1:cols));
    
    for i = 1:block_size:rows
        end_idx = min(i + block_size - 1, rows);
        X_sub = X(i:end_idx, :);
        cur_rows = end_idx - i + 1;
        
        Xs = sort(X_sub, 2, 'descend');
        Xsum = cumsum(Xs, 2);
        
        cond = Xs + (1 - Xsum) ./ idx_vec > 0;
        rho_idx = sum(cond, 2); 
        rho_idx = max(rho_idx, 1); 
        
        row_indices = gpuArray(int64((1:cur_rows)'));
        lin_idx = (int64(rho_idx) - 1) * int64(cur_rows) + row_indices;
        
        theta = (1 - Xsum(lin_idx)) ./ double(rho_idx);
        X_proj(i:end_idx, :) = max(X_sub + theta, 0);
    end
    
    if dim == 1, X_proj = X_proj'; end
end