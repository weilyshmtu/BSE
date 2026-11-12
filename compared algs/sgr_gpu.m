function [S, C, F, time_used, obj_val, acc_array1, acc_array2] = sgr_gpu(K, alpha, beta, opts)
% SGR: Structured Graph Regularized clustering (Mixed-Precision GPU Version)
% Hardware Optimized for NVIDIA RTX 4090 (24GB VRAM)
% Uses Single Precision for max speed and Double Precision for accurate Eigendecomposition.
tic

% --- 1. GPU Initialization & Memory Transfer (Single Precision) ---
n = opts.n;

K = gpuArray(single(K));
S_init_gpu = gpuArray(single(opts.S_init));
S = S_init_gpu;

F = gpuArray(single(opts.F_init));
FFt = F * F';
C = gpuArray.ones(n, n, 'single') / single(n);

% 
alpha = single(alpha);
beta = single(beta);
lambda = single(15);

iter = 1;
maxIter = 40;
stop_iter = 10;
tol1 = max(1e-3, 1/n);

% Initialize logging variables
obj_val = zeros(1, maxIter, 'single'); 
[obj_val(iter), ~, ~] = compute_obj_gpu(K, S, C, FFt, alpha, beta, lambda, opts);
acc_array1 = zeros(1, maxIter);
acc_array2 = zeros(1, maxIter);

mask_ref = S_init_gpu > 0; 

% --- 2. Main ADMM Loop ---
while iter < maxIter
    iter = iter + 1;
    
    % Update S (Single Precision Vectorized)
    [S, ~] = solve_S_gpu(K, S, C, alpha, beta, opts, mask_ref);
  
    % Update C (Single Precision Vectorized)
    [C, ~] = solve_C_gpu(S, C, FFt, alpha, beta, lambda, opts, mask_ref);
    
    % Update F (Mixed Precision: Double for eigs)
    [F, lambda, bflag] = solve_F_gpu(C, F, lambda, opts);
    FFt = F * F';
    
    % Update Objective
    [obj_val(iter), ~, ~] = compute_obj_gpu(K, S, C, FFt, alpha, beta, lambda, opts);
   
    obj_residual_t = abs(obj_val(iter) - obj_val(iter - 1)) / abs(obj_val(iter));
    obj_residual = gather(obj_residual_t); 
   
    if mod(iter, 10) == 0
        disp(['iter = ', num2str(iter), ', obj_res = ', num2str(obj_residual, '%6.4f'), ...
            ', acc1 = ', num2str(acc_array1(iter), '%6.4f'), ...
            ', acc2 = ', num2str(acc_array2(iter), '%6.4f')]) 
    end
    
    if iter > stop_iter
        if obj_residual < tol1 && bflag
            obj_val = gather(obj_val(1:iter));
            acc_array1 = acc_array1(1:iter);
            acc_array2 = acc_array2(1:iter);
            if bflag
               fprintf('the %d -th iteration -> end ...\n', iter)
            end
            time_used = toc;
            break;
        end             
    end
end

if iter >= maxIter
    time_used = toc;
    obj_val = gather(obj_val(1:iter));
end

% --- 3. Final Data Fetch ---
S = double(gather(S));
C = double(gather(C));
F = double(gather(F));
end

% =========================================================================
% Sub-function 1: solve_S_gpu
% =========================================================================
function [S, S_residual] = solve_S_gpu(K, S_pre, C, alpha, beta, opts, mask)
    n = opts.n;
    Y = gpuArray.zeros(n, n, 'single');
    U = S_pre;
    
    max_iter = 100;
    rho = single(1.5);
    mu = single(30);
    tol = 1e-5; 
    
    S_residual = zeros(1, max_iter, 'single');
    B_mat = K + 2 * beta * C; 
    
    for iter = 1:max_iter
        A_U = (alpha + beta) * U - alpha * (C * U);
        M_mat = U - (Y + A_U - B_mat) / mu;
        
        S = project_to_simplex_matrix_gpu(M_mat, mask);
        
        A_S = (alpha + beta) * S - alpha * (C * S);
        N_mat = S + (Y - A_S) / mu;
        U = (N_mat + N_mat') / 2;
        U(1:n+1:end) = 0; 
        
        Y = Y + mu * (S - U);
        mu = rho * mu;
        
        res = norm(S - S_pre, 'fro');
        S_residual(iter) = gather(res);
        if iter > 1 && res < tol
            break;
        else
            S_pre = S;
        end
    end
end

% =========================================================================
% Sub-function 2: solve_C_gpu
% =========================================================================
function [C, C_residual] = solve_C_gpu(S, C_pre, FFt, alpha, beta, lambda, opts, mask)
    n = opts.n;
    V = C_pre;
    Y = gpuArray.zeros(n, n, 'single');
    
    max_iter = 100;
    mu = single(30);
    rho = single(1.5);
    tol = 1e-5; % 放宽容差
    
    B_mat = alpha * (S * S') + 2 * beta * S + lambda * FFt;
    
    for iter = 1:max_iter
        A_V = beta * V;
        M_mat = V - (Y + A_V - B_mat) / mu;
        
        C = project_to_simplex_matrix_gpu(M_mat, mask);
    
        M = C + Y / mu;
        V = (M + M') / 2;
        V(1:n+1:end) = 0; 
        
        Y = Y + mu * (C - V);
        mu = mu * rho;
        
        res = norm(C - C_pre, 'fro');
        C_residual(iter) = gather(res);
        if iter > 1 && res < tol
            break;
        else
            C_pre = C;
        end
    end
end

% =========================================================================
% Sub-function 3: solve_F_gpu (Mixed Precision Core)
% =========================================================================
function [F, lambda, bflag] = solve_F_gpu(C, F, lambda, opts)
    bflag = false;
    c = opts.n_clusters;
    zr = 1e-10;
    F_old = F;
    
    C = (C + C') * single(0.5);
    
    L_C = -C;
    L_C(1:opts.n+1:end) = L_C(1:opts.n+1:end) + sum(C, 2)'; 
    
    L_C_double = double(L_C);
    
    try
        [V_sub, D_sub] = eigs(L_C_double, c+1, 'smallestreal');
    catch
        [V_sub, D_sub] = eigs(gather(L_C_double), c+1, 'smallestreal');
        V_sub = gpuArray(V_sub);
        D_sub = gpuArray(D_sub);
    end
    
    F = single(real(V_sub(:, 1:c)));
    ev = abs(diag(D_sub));
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    t = single(2);
    
    if fn1 > zr
        lambda = t * lambda;
    elseif fn2 < zr
        lambda = lambda / t;  
        F = F_old;
    else
        bflag = true;
    end 
end

% =========================================================================
% Sub-function 4: compute_obj_gpu
% =========================================================================
function [obj, obj1, obj2] = compute_obj_gpu(K, S, C, FFt, alpha, beta, lambda, opts)
    trace_K = sum(diag(K));
    term1_K = trace_K - sum(sum(K .* S')); 
    
    CS = C * S;
    term2_alpha = sum(S.^2, 'all') - sum(sum(S .* CS)); 
    
    term3_beta = beta * norm(S - C, 'fro')^2;
    
    obj1 = term1_K + alpha * term2_alpha + term3_beta;
    
    trace_FFt = sum(diag(FFt));
    obj2 = lambda * (trace_FFt - sum(sum(FFt .* C')));
    
    obj = obj1 + obj2;
end

% =========================================================================
% Helper: GPU Vectorized Masked Simplex Projection (Single Precision)
% =========================================================================
function X_proj = project_to_simplex_matrix_gpu(X, mask)
    X(~mask) = single(-1e10); 
    
    [d, n] = size(X);
    X_proj = gpuArray.zeros(d, n, 'single'); 
    
    block_size = 3000; 
    idx = gpuArray(single((1:d)'));
    
    for i = 1:block_size:n
        end_idx = min(i + block_size - 1, n);
        X_sub = X(:, i:end_idx);
        
        Xs = sort(X_sub, 1, 'descend');
        Xsum = cumsum(Xs, 1);
        
        cond = (Xs + (1 - Xsum) ./ idx) > 0;
        
        rho_idx = double(sum(cond, 1));
        rho_idx = max(rho_idx, 1); 
        
        cols = end_idx - i + 1;
        col_offsets = gpuArray(double(0:cols-1)) * double(d); 
        lin_idx = rho_idx + col_offsets;
        
        theta = (1 - Xsum(lin_idx)) ./ single(rho_idx);
        X_proj(:, i:end_idx) = max(X_sub + theta, 0);
    end
    
    X_proj(~mask) = 0;
end