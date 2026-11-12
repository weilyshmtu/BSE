function [G_final, time_used] = tgl_gpu(lambda1, lambda2, opts)
    % TGL: Twin Graph Learning (Optimized GPU implementation)
    % This version balances speed (single precision) with numerical 
    % stability (double precision for spectral updates).
    
    tic;
    n = opts.n;
    
    % --- VRAM-Optimized Input Handling ---
    Dx = gpuArray(single(full(opts.Dx)));
    XtX = gpuArray(single(full(opts.XtX)));
    
    % Initialize graphs and auxiliary variables
    Gn = gpuArray.zeros(n, n, 'single');
    Gs = gpuArray.zeros(n, n, 'single');
    P  = gpuArray.eye(n, 'single'); 
    
    Cn = gpuArray.zeros(n, n, 'single'); 
    Cs = gpuArray.zeros(n, n, 'single');
    
    Y   = gpuArray.zeros(n, n, 'single');    
    Wn  = gpuArray.zeros(n, n, 'single'); 
    Ws  = gpuArray.zeros(n, n, 'single');
    
    % Parameters
    mu = single(1e-4); 
    rho = single(1.1); 
    max_mu = single(1e6);
    
    maxIter = 100;
    tol = 1e-6;
    
    % Rank for rSVD (Adaptive to cluster count)
    rsvd_k = 1000; 
    if isfield(opts, 'k'), rsvd_k = opts.k; end
    
    % Precompute constant part for P update
    inv_part = 2 * single(lambda2) * XtX;
    
    for iter = 1:maxIter
        % 1. Update Gn (Neighbor Graph via Simplex Projection)
        Qn = Cn - (1/mu) * Wn - (single(lambda1)/mu) * Dx;
        
        if any(isnan(Qn), 'all') || any(isinf(Qn), 'all')
            warning('TGL_GPU: 迭代第 %d 次时在 Qn 中检测到 NaN/Inf，数值发散，已提前终止迭代。', iter);
            break;
        end
        Gn = project_to_simplex_matrix_gpu(Qn); 
        
        % 2. Update Gs (Self-expressiveness Graph)
        Qs = 0.5 * (P + Cs) - (1/(2*mu)) * (Y + Ws);
        
        % 【新增拦截机制】：捕捉 Qs 中的 NaN 或 Inf
        if any(isnan(Qs), 'all') || any(isinf(Qs), 'all')
            warning('TGL_GPU: 迭代第 %d 次时在 Qs 中检测到 NaN/Inf，数值发散，已提前终止迭代。', iter);
            break;
        end
        Gs = project_to_simplex_matrix_gpu(Qs);
        
        % 3. Update P (Numerical stabilization using double precision casting)
        LHS = inv_part;
        LHS(1:n+1:end) = LHS(1:n+1:end) + mu; 
        RHS = inv_part + Y + mu * Gs;
        P = double(LHS) \ double(RHS); % Solve linear system in double precision
        P = single(P);
        
        % 4. Update C_n, C_s (2-Point Math Trick replacing 3D Tensor FFT)
        X1 = (Gn + (1/mu)*Wn) + (Gs + (1/mu)*Ws);
        X2 = (Gn + (1/mu)*Wn) - (Gs + (1/mu)*Ws);
        
        % Apply thresholding in spectral domain via rSVD
        X1_new = rsvd_svt_gpu(X1, 1/mu, rsvd_k);
        X2_new = rsvd_svt_gpu(X2, 1/mu, rsvd_k);
        
        Cn = 0.5 * (X1_new + X2_new);
        Cs = 0.5 * (X1_new - X2_new);
        
        % 5. Update Multipliers
        Y = Y + mu * (Gs - P);
        Wn = Wn + mu * (Gn - Cn);
        Ws = Ws + mu * (Gs - Cs);
        mu = min(mu * rho, max_mu);
        
        % 6. Convergence Check
        err = max(norm(Gs - P, 'fro')/n, max(norm(Gn - Cn, 'fro')/n, norm(Gs - Cs, 'fro')/n));
        if err < tol, break; end
    end
    
    G_final = gather(0.25 * (Gn + Gn' + Gs + Gs'));
    time_used = toc;
end

function X_new = rsvd_svt_gpu(X, kappa, k)
    % Efficient GPU Randomized SVD with Soft-Thresholding
    n = size(X, 1);
    % Add epsilon regularization to improve conditioning
    X_reg = X;
    X_reg(1:n+1:end) = X_reg(1:n+1:end) + 1e-4;
    
    % Randomized projection
    Omega = randn(n, k, 'single', 'gpuArray');
    Y = X_reg * Omega;
    [Q, ~] = qr(Y, 0); 
    
    % Precise SVD on the reduced subspace
    B = Q' * X_reg;
    [U_tilde, S, V] = svd(double(B), 'econ'); % Double precision for spectral stability
    
    % Soft-thresholding on singular values
    s = max(diag(S) - double(kappa), 0);
    
    % Reconstruct
    X_new = single(Q * (U_tilde * diag(s) * V'));
end

function S = project_to_simplex_matrix_gpu(V)
    % Optimized simplex projection for large matrices
    [n, m] = size(V);
    S = gpuArray.zeros(n, m, 'single');
    block_size = 2000; 
    
    for i = 1:block_size:m
        end_idx = min(i + block_size - 1, m);
        V_sub = V(:, i:end_idx);
        V_sorted = sort(V_sub, 1, 'descend');
        V_cum = cumsum(V_sorted, 1);
        
        rho = sum(V_sorted - (V_cum - 1) ./ (1:n)' > 0, 1);
        
        rho = max(rho, 1);
        
        theta = (V_cum(sub2ind([n, end_idx-i+1], rho, 1:size(V_sub, 2))) - 1) ./ rho;
        
        S(:, i:end_idx) = max(V_sub - theta, 0);
    end
end