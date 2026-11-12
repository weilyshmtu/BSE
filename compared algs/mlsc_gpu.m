function [F_star, G_star, time_used] = mlsc_gpu(X, alpha, beta, opts)
    % MLSC_GPU: Multi-view Latent Subspace Clustering (Optimized GPU Version)
    % Optimized for RTX 4090D: Keeps representation layers in fast FP32 while
    % isolating FP64 strictly to the sensitive linear system solvers.
    
    tic
    % Use single precision for the input matrix to maximize VRAM bandwidth
    X = gpuArray(single(full(X))); 
    d = opts.d; n = opts.n;
    
    % Keep penalty and regularization scalars in double precision
    alpha_d = double(alpha);
    beta_d = double(beta);
    mu = double(1e-6);
    
    I_n = gpuArray.eye(n, 'single');
    
    % --- Precomputations ---
    % [PERFORMANCE FIX]: Compute centered data once outside the loop
    X_centered = X - mean(X, 1); 
    
    % --- Initialization ---
    G = gpuArray.zeros(n, n, 'single');
    H = gpuArray.zeros(n, n, 'single');
    F = gpuArray.zeros(n, n, 'single');
    W1 = gpuArray.zeros(n, n, 'single');
    W2 = gpuArray.zeros(n, n, 'single');
    P = gpuArray(randn(50, d, 'single') * 0.01);
    
    opts.max_iter = 100;
    opts.v = 1.2;
    opts.max_mu = 1e8;
    opts.eps = 1e-8; 
    opts.eta = 1e-5;
    
    for iter = 1:opts.max_iter
        
        % 1. Update P (Subspace Projection Layer)
        % [PERFORMANCE FIX]: Entirely calculated in FP32 to unleash 4090D tensor speed
        U = P * X_centered;
        norms_U = sqrt(sum(U.^2, 1)) + single(eps('single'));
        U_hat = U ./ norms_U;
        S_cos = U_hat' * U_hat;
        Dist2 = max(0, 1 - S_cos);
        V = U_hat * G;
        d_vec = sum(U_hat .* V, 1);
        grad_U = -2 * (V - U_hat .* d_vec) ./ norms_U;
        grad_P = grad_U * X_centered';
        P = P - single(opts.eta) * grad_P;
        
        % 2. Update F
        HtH = H' * H; 
        LHS_F = double(mu) * double(HtH);
        
        % [MATHEMATICAL FIX]: Correctly add 2*beta directly to the diagonal 
        % instead of modifying scales outside the backslash solver
        LHS_F(1:n+1:end) = LHS_F(1:n+1:end) + (2 * beta_d); 
        RHS_F = double(mu) * double(HtH) + double(H' * W1);
        
        F = single(LHS_F \ RHS_F);
        
        % 3. Update H
        I_F = I_n - F;
        IF_t = I_F';
        IF_IFt = I_F * IF_t; 
        
        A_H = single(mu) * G + W2 - W1 * IF_t;
        
        LHS_H = double(mu) * double(IF_IFt);
        LHS_H(1:n+1:end) = LHS_H(1:n+1:end) + (2 * alpha_d + double(mu));
        
        % Solve system symmetrically via right-division in double precision
        H = single( double(A_H) / LHS_H );
        
        % --- NUMERICAL SAFETY INTERCEPT ---
        if any(isnan(H), 'all') || any(isinf(H), 'all')
            warning('MLSC_GPU: NaN/Inf detected at iteration %d. Terminating prematurely to preserve state.', iter);
            break; 
        end
        
        % 4. Update G: Simplex projection
        G_input = (-1 / single(mu)) * (Dist2 - single(mu) * H + W2);
        G = project_to_simplex_matrix_mixed( double(G_input) );
        
        % 5. Update multipliers & mu
        W1 = W1 + single(mu) * (H - H * F);
        W2 = W2 + single(mu) * (G - H);
        mu = min(double(opts.v) * mu, double(opts.max_mu));
        
        % 6. Convergence check
        if mod(iter, 10) == 0
            if norm(H - H * F, 'fro')^2 < opts.eps && norm(G - H, 'fro')^2 < opts.eps
                break;
            end
        end
    end
    
    % --- Final aggregation ---
    F_star = gather(double(F));
    G_star = gather(double(G));
    time_used = toc;
end

function X_proj = project_to_simplex_matrix_mixed(X)
    % Performs simplex projection. Double precision maintained internally for accumulation.
    [n, d] = size(X);
    [Xs, ~] = sort(X, 2, 'descend');
    Xsum = cumsum(Xs, 2);
    
    % [OPTIMIZATION]: Pre-cast index vectors to GPU to avoid implicit host-to-device transfers
    d_idx = gpuArray(double(1:d));
    rho = sum(Xs + (1 - Xsum) ./ d_idx > 0, 2);
    rho = max(rho, 1);
    
    % Fast linear indexing using column-major mapping
    row_idx = gpuArray(double((1:n)'));
    lin_idx = (rho - 1) .* n + row_idx;
    theta = (1 - Xsum(lin_idx)) ./ rho;
    
    X_proj = single(max(X + theta, 0));
end