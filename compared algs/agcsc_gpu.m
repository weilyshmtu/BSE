function [C, F, time_used] = agcsc_gpu(X, alpha, beta, opts)
    % AGCSC: Adaptive Graph Clustering (GPU Mixed Precision + Stable ADMM)
    % Optimized for RTX 4090D 
    
    tic;
    
    % --- 1. GPU Memory Setup (Single precision for data to save VRAM/Bandwidth) ---
    X_s = gpuArray(single(full(X)));
    XXt_s = gpuArray(single(full(opts.XtX)));
    n = opts.n;
    maxIter = opts.max_iter;
    I_s = gpuArray.eye(n, 'single');
    
    alpha_d = double(alpha);
    beta_d = double(beta);
    
    % --- 2. Stabilized ADMM Parameters ---
    tol = 1e-5;
    rho = 1.05; % Reduced from 1.1 to prevent aggressive divergence
    max_mu = 1e10; % Capped at 1e8. 1e30 will destroy any matrix solver
    
    % [CRITICAL FIX] Data-scale aware mu initialization
    % Prevents mu from being absorbed by floating-point truncation when XXt is large
    scale_XXt = double(max(abs(XXt_s(:))));
    mu = min(max(1e-6, scale_XXt * 1e-6), 1e-2); 
    
    % Initialize Variables
    F_s = X_s;
    Z_s = gpuArray.zeros(n, n, 'single');
    
    % Dual variables MUST be in double precision to prevent long-term round-off error
    Y1_d = gpuArray.zeros(n, n, 'double'); 
    Y2_d = gpuArray.zeros(n, 1, 'double');
    
    iter = 0;
    while iter < maxIter
        iter = iter + 1;
        
        % =====================================================================
        % 1. Update C (Mixed Precision + Implicit Expansion)
        % =====================================================================
        FFt_s = F_s * F_s';
        I_Z_s = I_s - Z_s;
        IZ_IZt_s = I_Z_s * I_Z_s';
        
        % Assemble A in double precision
        A_d = 2.0 * double(XXt_s) + (2.0 * alpha_d) * double(FFt_s) + (2.0 * beta_d) * double(IZ_IZt_s);
        A_d = A_d + mu;                        % Equivalent to + mu*eet
        A_d(1:n+1:end) = A_d(1:n+1:end) + mu;  % Equivalent to + mu*I
        A_d = 0.5 * (A_d + A_d');              % Safeguard: Force exact symmetry
        
        % Assemble B in double precision
        B_d = 4.0 * double(F_s * X_s') - 2.0 * double(XXt_s) + (2.0 * alpha_d) * double(X_s * F_s') + mu * double(Z_s) - Y1_d;
        B_d = B_d + mu;               
        B_d = B_d - Y2_d; % Implicit expansion: mathematically identical to ( - Y2 * e' )
        
        % Solve C (B/A is equivalent to (A\B')' for symmetric A)
        C_d = (A_d \ B_d')';
        C_s = single(C_d); % Cast to FP32 for downstream heavy math
        
        % =====================================================================
        % 2. Update F (Heavy math in FP32, solver in FP64)
        % =====================================================================
        CtC_s = C_s' * C_s;
        
        A_F_d = alpha_d * double(CtC_s) + 4.0 * double(I_s);
        A_F_d = 0.5 * (A_F_d + A_F_d');
        A_F_d(1:n+1:end) = A_F_d(1:n+1:end) + eps;
        
        B_F_d = 2.0 * double(C_s * X_s) + 2.0 * double(X_s) + alpha_d * double(C_s' * X_s);
        F_s = single(A_F_d \ B_F_d);
        
        % =====================================================================
        % 3. Update Z (FP64 Solver)
        % =====================================================================
        A_Z_d = (2.0 * beta_d) * double(CtC_s) + mu * double(I_s);
        A_Z_d = 0.5 * (A_Z_d + A_Z_d');
        A_Z_d(1:n+1:end) = A_Z_d(1:n+1:end) + eps;
        
        B_Z_d = (2.0 * beta_d) * double(CtC_s) + Y1_d + mu * double(C_s);
        Z_d = A_Z_d \ B_Z_d;
        
        % Constraints
        Z_d(1:n+1:end) = 0;
        Z_d = 0.5 * (Z_d + Z_d');
        Z_s = single(max(Z_d, 0.0));
        
        % =====================================================================
        % 4. Convergence Check & Multiplier Update
        % =====================================================================
        leq1_d = C_d - double(Z_s);
        leq2_d = sum(C_d, 2) - 1.0; % sum(..., 2) is mathematically identical to C*e
        
        stopC_gpu = max(max(max(abs(leq1_d))), max(abs(leq2_d)));
        stopC = gather(stopC_gpu);
        
        if iter == 1 || mod(iter, 50) == 0 || stopC < tol
            fprintf('iter %d, mu=%2.1e, stopC=%2.3e\n', iter, mu, stopC);
        end
        
        % NaN escape hatch to prevent system freeze
        if stopC < tol || isnan(stopC) 
            break;
        else
            % Update multipliers safely in double precision
            Y1_d = Y1_d + mu * leq1_d;
            Y2_d = Y2_d + mu * leq2_d;
            mu = min(max_mu, mu * rho);
        end
    end
    
    C = gather(C_s);
    F = gather(F_s);
    time_used = toc;
end