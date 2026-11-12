function [C2, F, W, time_used] = isc_gpu(Y, F, gamma, lambda, opts)
%--------------------------------------------------------------------------
% HIGH-PERFORMANCE ISC-ADMM (FP32, Precomputations & Memory Boundaries)
% 
% Optimizations applied:
% 1. Global Single Precision (FP32) to halve memory bandwidth pressure.
% 2. Division-to-Multiplication conversion to save GPU ALU cycles.
% 3. Strict explicit symmetric definitions to trigger fast cuBLAS solves.
% 4. Optimized CPU-GPU IO bindings for the LBFGSB solver.
%--------------------------------------------------------------------------
tic; 
maxIter = 50;
beta_s = single(1e-3);
alpha_s = single(gamma);
d = opts.d;
N = opts.n;

% --- Optimization 1: Strictly FP32 Initialization ---
Y_gpu = gpuArray(single(full(Y)));
F_gpu = gpuArray(single(full(F)));
W = gpuArray.ones(d, N, 'single'); 

% Precompute scalar parameters as single
lambda_s = single(lambda);
lambda_max = computeLambda_mat_gpu(Y_gpu);
mu1 = lambda_s * (single(1) / lambda_max);
mu2 = lambda_s * single(1);

inv_mu2 = single(1) / mu2; % Div-to-mul optimization
ratio1 = (single(2) * mu1) * inv_mu2;
ratio2 = beta_s * inv_mu2;

C1 = gpuArray.zeros(N, N, 'single');
Lambda2 = gpuArray.zeros(N, N, 'single');
Lambda3 = gpuArray.zeros(1, N, 'single');
ones_N = gpuArray.ones(N, 1, 'single'); 
I_d = gpuArray.eye(d, 'single');
I_N = gpuArray.eye(N, 'single');

i = 1;

% Normalize F and compute initial distance matrix Q
F_norm = F_gpu ./ (sqrt(sum(F_gpu.*F_gpu, 1)) + eps('single')); 
Q = L2_distance_gpu(F_norm, F_norm);

while (i < maxIter)
    YW = Y_gpu .* W;
    
    % -----------------------------------------------------------------
    % Updating Z (Representation Matrix)
    % -----------------------------------------------------------------
    if d < N
        epsilon = single(2)*mu1*inv_mu2;
        
        % --- Optimization 2: Woodbury Associative Law Optimization ---
        inv_mat = inv(I_d + epsilon * (YW * YW')); 
        
        % Precompute common term2
        term2 = ratio1*(YW'*Y_gpu) + (C1 - Lambda2*inv_mu2) + ones_N*(ones_N' - Lambda3*inv_mu2) - ratio2*Q;
        
        % Right-associative multiplication to avoid N x N expansions
        temp_sum_J = ones_N - epsilon * (YW' * (inv_mat * (YW * ones_N))); 
        J_term2 = term2 - epsilon * (YW' * (inv_mat * (YW * term2))); 
        
        scalar_val = single(1) + sum(temp_sum_J);
        Z = J_term2 - temp_sum_J * ((temp_sum_J' * term2) / scalar_val);
        
    else
        % Non-Woodbury Fallback
        LHS = single(2)*mu1*(YW'*YW) + mu2*I_N + mu2*(ones_N * ones_N');
        RHS = single(2)*mu1*(YW'*Y_gpu) + mu2*(C1 - Lambda2*inv_mu2) + mu2*ones_N*(ones_N' - Lambda3*inv_mu2);
        
        % Force symmetry to trigger fast cuBLAS symmetric solvers
        LHS = single(0.5) * (LHS + LHS'); 
        Z = (LHS \ RHS) - beta_s*Q;
    end
    Z(1:N+1:end) = single(0); 
    
    % -----------------------------------------------------------------
    % Updating C2 via Soft-Thresholding
    % -----------------------------------------------------------------
    temp_C = Z + Lambda2*inv_mu2;
    C2 = max(single(0), abs(temp_C) - inv_mu2) .* sign(temp_C);
    C2(1:N+1:end) = single(0);
    
    % -----------------------------------------------------------------
    % Updating W (Weight Matrix)
    % -----------------------------------------------------------------
    opts.grad = 1;
    if opts.grad
        W = LBFGSB_W_gpu(Y_gpu, F_gpu, C2, W, lambda_s, alpha_s, alpha_s); % gamma is passed as alpha based on init
    end    
    
    % -----------------------------------------------------------------
    % Updating F (Cluster Indicator Matrix)
    % -----------------------------------------------------------------
    opts.updateY = 1;
    if opts.updateY
        % Avoid diag() allocations of N x N matrices
        L_mat = - (beta_s / single(2)) * (C2 + C2');
        sum_C2 = sum(C2, 1);
        L_mat(1:N+1:end) = L_mat(1:N+1:end) + beta_s * sum_C2;
        
        S2_vec = sum(W, 1);
        S2 = S2_vec' * S2_vec;
        
        [F_gpu, ~] = solve_Y_gpu(L_mat, alpha_s * S2, F_gpu);
        
        F_norm = F_gpu ./ (sqrt(sum(F_gpu.*F_gpu, 1)) + eps('single'));
        Q = L2_distance_gpu(F_norm, F_norm);
    end
    
    % Multipliers Update
    Lambda2 = Lambda2 + mu2 * (Z - C2);
    Lambda3 = Lambda3 + mu2 * (ones_N' * Z - ones_N');  
    C1 = C2;
    i = i + 1;    
end

% --- Step 2: Gather results back to CPU ---
C2 = double(gather(C2)); 
F = double(gather(F_gpu)); 
W = double(gather(W));
time_used = toc;

end

%% --- SUBFUNCTIONS (GPU ACCELERATED & VRAM OPTIMIZED) ---

function lambda = computeLambda_mat_gpu(Y, P)
    if (nargin < 2), P = Y; end
    N = size(Y, 2);
    T = P' * Y;
    T(1:N+1:end) = single(0); 
    T = abs(T);
    lambda = min(max(T, [], 1));
end

function [W_result] = LBFGSB_W_gpu(X, F, S, W_init, lambda, gamma, alpha)
    [d, n] = size(W_init);
    
    % Maintain strict Single Precision during heavy pre-computations on GPU
    XXS = single(2) .* X .* (X * S');
    SS = S * S';
    YY = F' * F;
    
    funObj = @(x) ISC_obj_W_gpu(X, F, S, XXS, SS, YY, lambda, alpha, gamma, n, d, x);
    
    % CPU Solver strictly mandates double precision; gather only the lightweight vector
    x_init = double(gather(reshape(W_init, d*n, 1))); 
    LB = zeros(d*n, 1, 'double');
    UB = ones(d*n, 1, 'double');
    
    opts_lbfgs = [];
    opts_lbfgs.x0 = x_init;
    opts_lbfgs.pgtol = 1e-3;
    opts_lbfgs.maxIts = 1;
    opts_lbfgs.printEvery = -1;
    
    x = ISC_lbfgsb(funObj, LB, UB, opts_lbfgs); % This calls your external solver
    
    % Push back to GPU and cast to single
    W_result = gpuArray(single(reshape(x, d, n)));
end

function [obj, grad] = ISC_obj_W_gpu(X, F, S, XXS, SS, YY, lambda, alpha, gamma, n, d, x)
    % Cast incoming double data back to single precision for fast GPU calculation
    W = gpuArray(single(reshape(x, d, n)));
    
    term1 = X - (X .* W) * S;
    obj_term1 = lambda * sum(sum(term1.^2));
    
    w_tmp = sum(W, 1);
    obj_term2 = alpha * sum(w_tmp.^2);
    
    term3 = w_tmp * YY; 
    obj_term3 = gamma * (term3 * w_tmp');
    
    % Return double obj to CPU
    obj = double(gather(obj_term1 + obj_term2 + obj_term3)); 
    
    % Fully Vectorized Gradient Calculation on GPU
    grad1 = (single(2)*lambda) .* X .* ((X .* W) * SS) - XXS;     
    grad2 = (single(2)*alpha) .* w_tmp; 
    grad3 = (single(2)*gamma) .* (w_tmp * YY); 
    
    grad = grad1 + grad2 + grad3; 
    grad = double(gather(reshape(grad, d*n, 1))); 
end

function [F, obj] = solve_Y_gpu(K, A, F)
    % Discrete optimization: Kept logically identical but data type matched
    fKf = sum(F .* (K * F')')';
    ff = sum(F, 1)';
    fAf = sum(F .* (A * F')')';
    [~, m_all] = max(F, [], 2);
    n_samples = size(F, 1);
    
    for i = 1:n_samples
        m = m_all(i);
        if ff(m) <= single(1), continue; end 
    
        Y_F = F' * K(:, i);
        Y_A = F' * A(:, i);
        
        fKf_s = fKf + single(2) * Y_F + K(i, i); fKf_s(m) = fKf(m);
        ff_k = ff + single(1); ff_k(m) = ff(m);
        fAf_s = fAf + single(2) * Y_A + A(i, i); fAf_s(m) = fAf(m);
    
        fKf_0 = fKf; fKf_0(m) = fKf(m) - single(2) * Y_F(m) + K(i, i);
        ff_0 = ff; ff_0(m) = ff(m) - single(1);
        fAf_0 = fAf; fAf_0(m) = fAf(m) - single(2) * Y_A(m) + A(i, i);
    
        delta = fKf_s ./ ff_k - fKf_0 ./ ff_0 + fAf_s - fAf_0;
        [~, p] = min(delta);
        
        if p ~= m
            fKf([m, p]) = [fKf_0(m), fKf_s(p)];
            fAf([m, p]) = [fAf_0(m), fAf_s(p)];
            ff([m, p]) = [ff_0(m), ff_k(p)];
            F(i, [p, m]) = [single(1), single(0)];
            m_all(i) = p;
        end
    end
    obj = sum(fKf ./ ff);
end

function D = L2_distance_gpu(a, b)
    if nargin < 2, b = a; end
    aa = sum(a.*a, 1); 
    bb = sum(b.*b, 1);
    ab = a' * b;
    % Uses modern MATLAB implicit expansion
    D = max(single(0), aa' + bb - single(2) * ab); 
end