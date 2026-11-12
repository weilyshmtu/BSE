function [S, F, time_used] = ercan_gpu(X, k, eta, opts)
    % ERCAN: Entropy Regularization for Unsupervised Clustering with Adaptive Neighbors
    % Optimized for RTX 4090D: Fixed NaN/Inf crashes, vectorized operations, 
    % and synchronized mathematical parameter updates.
    
    tic;
    n = opts.n;
    c = opts.n_clusters;
    
    % Use single precision to save VRAM
    Dx = gpuArray(single(opts.Dx)); 
    fro_norm_Dx = norm(Dx, 'fro');
    gamma = (eta / n) * fro_norm_Dx;
    lambda = gamma;
    
    % Initialization of S (Vectorized & Safe)
    Dx_temp = Dx;
    Dx_temp(1:n+1:end) = inf; % Mask the main diagonal only
    
    % SAFE SOFTMAX: Prevent underflow to 0 which leads to NaN
    val = -Dx_temp / gamma;
    val = val - max(val, [], 2); 
    S_raw = exp(val);
    S = S_raw ./ sum(S_raw, 2);
    
    % Spectral initialization of F (Force double precision for eigenvalue decomposition)
    S_sym = (S + S') / 2;
    L_s = diag(sum(S_sym, 2)) - S_sym;
    L_s = 0.5 * (L_s + L_s'); % Ensure exact symmetry
    [F, ~, ~] = eig1_gpu(double(L_s), c, 0);
    
    maxIter = 100;
    tol = 1e-10; % Original convergence tolerance
    block_size = 2000; 
    
    for iter = 1:maxIter
        % 1. Update F
        S_sym = (S + S') / 2;
        D_vec = sum(S_sym, 2);
        L_s = diag(D_vec) - S_sym;
        L_s = 0.5 * (L_s + L_s'); 
        F_old = F;
        
        [F, ~, ev] = eig1_gpu(double(L_s), c, 0);
        
        % [Robustness Fix] Filter out negative noise from GPU floating-point precision
        ev = max(0, ev); 
        
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1));
        
        % Simplified connectivity check to minimize memory overhead
        flag = sum(D_vec - 1 < 0.01); 
        
        % Dynamic eta adjustment logic
        if fn1 > tol
            eta = 2 * eta;
            % [CRITICAL]: Must re-calculate gamma & lambda when eta changes
            gamma = (eta / n) * fro_norm_Dx;
            lambda = gamma;
        elseif fn2 < tol
            eta = eta / 2;
            gamma = (eta / n) * fro_norm_Dx;
            lambda = gamma;
            F = F_old;
        elseif fn1 < tol && fn2 > tol && flag == n
            break;
        end
        
        % 2. Update S (Block-wise calculation to prevent OOM)
        F_c = F(:, 1:c);
        sumF = sum(F_c.^2, 2);
        
        S = gpuArray.zeros(n, n, 'single');
        
        for i = 1:block_size:n
            end_idx = min(i + block_size - 1, n);
            chunk_size = end_idx - i + 1;
            
            % Compute distances locally
            sumF_chunk = sumF(i:end_idx);
            Df_chunk = sumF_chunk + sumF' - 2 * (F_c(i:end_idx, :) * F_c');
            
            D_all_chunk = Dx(i:end_idx, :) + lambda * max(single(0), Df_chunk);
            
            % Set only the block's diagonal to Inf
            lin_idx_diag = (i:end_idx)' - 1; 
            lin_idx_diag = lin_idx_diag * chunk_size + (1:chunk_size)';
            D_all_chunk(lin_idx_diag) = inf;
            
            % Local sorting and selection
            [sorted_D, idx_D] = sort(D_all_chunk, 2, 'ascend');
            
            D_k = sorted_D(:, 1:k);
            idx_k = idx_D(:, 1:k);
            
            % Safe Softmax to prevent division by zero
            val = -D_k / gamma;
            val = val - max(val, [], 2); 
            s_vals = exp(val);
            s_vals = s_vals ./ sum(s_vals, 2);
            
            % [PERFORMANCE FIX]: Vectorized assignment instead of slow loops
            row_idx = repmat((1:chunk_size)', 1, k);
            global_row = i + row_idx - 1;
            lin_S = (idx_k - 1) * n + global_row; 
            
            S(lin_S) = s_vals;
        end
    end
    
    time_used = toc;
    S = gather(S);
    F = gather(F);
end

%% --- Helper Function ---
function [V, D, evs] = eig1_gpu(L, c, is_smallest)
    % Robust eigenvalue solver with correction for numerical stability
    L = (L + L') / 2; 
    n = size(L, 1);
    k_eigs = min(c + 1, n); 
    
    opts_eigs.disp = 0;
    opts_eigs.tol = 1e-5; 
    
    if is_smallest == 0
        sigma = -1e-5; 
        sort_dir = 'ascend';
        % Apply small shift to prevent singularity in eigenvalue decomposition
        shift_val = 1e-4; 
        L(1:n+1:end) = L(1:n+1:end) + shift_val; 
    else
        sigma = 'largestabs';
        sort_dir = 'descend';
        shift_val = 0;
    end
    
    try
        [V_all, D_all] = eigs(L, k_eigs, sigma, opts_eigs);
    catch
        % Fallback to CPU if GPU processing fails
        L_cpu = double(gather(L));
        [V_all, D_all] = eigs(L_cpu, k_eigs, sigma, opts_eigs);
        V_all = gpuArray(V_all);
        D_all = gpuArray(D_all);
    end
    
    evs_all = diag(D_all);
    
    if is_smallest == 0
        % Remove the artificial shift to restore true eigenvalues
        evs_all = evs_all - shift_val; 
        % Force tiny floating-point noise to zero
        evs_all(abs(evs_all) < 1e-8) = 0; 
    end
    
    [evs, i] = sort(evs_all, sort_dir);
    
    V = V_all(:, i(1:c));
    D = evs(1:c);
end