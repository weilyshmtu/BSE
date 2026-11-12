function [Z, E, time_used] = lrr_gpu(X, lambda, opts)
% LRR: Low-Rank Representation (Robust GPU Version)
% This version includes a CPU-fallback mechanism and numerical clamping 
% to solve the "SVD did not converge" issue on GPU.

tic;
tol = 1e-6;
maxIter = 1000; 
rho = 1.1;
max_mu = 1e8;   
mu = 1e-2;

d = opts.d;
n = opts.n;

% --- Pre-processing: Data Normalization (Crucial for GPU Stability) ---
% If X has very large values, X'*X will explode. 
% We check the norm and scale if necessary.
x_norm = norm(X, 'fro');
if x_norm > 1e4
    X = X / (x_norm / 100); 
end

% Move data to GPU
X = gpuArray(X);
xtx = gpuArray(opts.XtX);
I_n = gpuArray.eye(n);
inv_x = (xtx + I_n) \ I_n;

% Initializing variables
J = gpuArray.zeros(n, n);
Z = gpuArray.zeros(n, n);
E = gpuArray.zeros(d, n);
Y1 = gpuArray.zeros(d, n);
Y2 = gpuArray.zeros(n, n);

iter = 0;
while iter < maxIter
    iter = iter + 1;
    
    % --- Update J (Singular Value Thresholding with Failsafe) ---
    temp = Z + Y2/mu;
    
    % 1. Numerical Clamping: Prevent Inf/NaN from entering SVD
    temp(isnan(temp)) = 0;
    temp(isinf(temp)) = sign(temp(isinf(temp))) * 1e10;
    
    try
        % Attempt GPU SVD
        [U, sigma_mat, V] = svd(temp, 'econ');
    catch
        % 2. CPU Fallback: If GPU SVD fails, compute on CPU and move back
        fprintf('Iteration %d: GPU SVD failed to converge. Falling back to CPU...\n', iter);
        temp_cpu = gather(temp);
        [U_cpu, sigma_cpu, V_cpu] = svd(temp_cpu, 'econ');
        U = gpuArray(U_cpu);
        sigma_mat = gpuArray(sigma_cpu);
        V = gpuArray(V_cpu);
    end
    
    sigma = diag(sigma_mat);
    svp = sum(sigma > 1/mu);
    if svp >= 1
        sigma = sigma(1:svp) - 1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:, 1:svp) * diag(sigma) * V(:, 1:svp)';
    
    % --- Update Z ---
    term_Y = (X'*Y1 - Y2) / mu;
    Z = inv_x * (xtx - X'*E + J + term_Y);
    
    % --- Update E (L2,1-norm minimization) ---
    xmaz = X - X*Z;
    temp_E = xmaz + Y1/mu;
    E_norms = sqrt(sum(temp_E.^2, 1));
    scaling = max(0, 1 - (lambda/mu) ./ (E_norms + eps));
    E = temp_E .* scaling;
    
    % --- Convergence Check ---
    leq1 = xmaz - E;
    leq2 = Z - J;
    stopC = max(max(abs(leq1(:))), max(abs(leq2(:))));
    
    if iter == 1 || mod(iter, 50) == 0 || gather(stopC) < tol
        curr_err = gather(stopC);
        disp(['Iter ' num2str(iter) ', mu=' num2str(mu,'%2.1e') ', stopALM=' num2str(curr_err,'%2.3e')]);
        if curr_err < tol, break; end
    end
    
    % --- Update Multipliers ---
    Y1 = Y1 + mu * leq1;
    Y2 = Y2 + mu * leq2;
    
    % Adaptive mu: If error is still large, slow down mu growth
    if gather(stopC) > 10
        mu = min(max_mu, mu * 1.01); % Slow growth
    else
        mu = min(max_mu, mu * rho);
    end
end

Z = gather(Z);
E = gather(E);
time_used = toc;
end