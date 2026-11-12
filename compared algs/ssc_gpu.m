function [Z, time_used] = ssc_gpu(X, lambda, opts)
% SSC: Sparse Subspace Clustering (L1-norm with F-norm error)
% GPU Version: Fully vectorized for ADMM updates.

tic;
if nargin < 2
    lambda = 1;
end

% --- Parameters ---
tol = 1e-6;
maxIter = 1000; % Usually ADMM converges much earlier than 1e6
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

n = opts.n;
d = opts.d;

% --- Move Data to GPU ---
X = gpuArray(X);
xtx = gpuArray(opts.XtX);
I_n = gpuArray.eye(n);
inv_x = gpuArray(inv(opts.XtX + eye(n))); % Precomputed on CPU or GPU

% --- Initializing optimization variables on GPU ---
J = gpuArray.zeros(n, n);
E = gpuArray.zeros(d, n);
Z = J;

Y1 = gpuArray.zeros(d, n);
Y2 = gpuArray.zeros(n, n);

% --- Main loop ---
iter = 0;
while iter < maxIter
    iter = iter + 1;
    
    % --- Update J (Shrinkage operator / Soft-thresholding) ---
    temp_J = Z + Y2/mu;
    % J = soft_thresholding(temp_J, 1/mu)
    J = max(0, temp_J - 1/mu) + min(0, temp_J + 1/mu);
    % Enforce Jii = 0 (using linear indexing for speed on GPU)
    J(1:n+1:end) = 0; 
    
    % --- Update Z (Solve linear system) ---
    % Z = inv(X'X + I) * (X'X - X'E + J + (X'Y1 - Y2)/mu)
    B = xtx - X'*E + J + (X'*Y1 - Y2)/mu;
    Z = inv_x * B;
    
    % --- Update E (F-norm minimization) ---
    xmaz = X - X*Z;
    temp_E = xmaz + Y1/mu;
    E = (mu * temp_E) / (2 * lambda + mu);
    
    % --- Convergence Check ---
    leq1 = xmaz - E;
    leq2 = Z - J;
    
    % Compute stopC on GPU
    stopC = max(max(abs(leq1(:))), max(abs(leq2(:))));
    
    if iter == 1 || mod(iter, 50) == 0 || gather(stopC) < tol
        curr_stop = gather(stopC);
        disp(['iter ' num2str(iter) ', mu=' num2str(mu,'%2.1e') ', stopALM=' num2str(curr_stop,'%2.3e')]);
        if curr_stop < tol
            break;
        end
    end
    
    % --- Update Multipliers & Mu ---
    Y1 = Y1 + mu * leq1;
    Y2 = Y2 + mu * leq2;
    mu = min(max_mu, mu * rho);
end

time_used = toc;

% Gather results back to CPU
Z = gather(Z);
end