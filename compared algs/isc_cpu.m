function [C2, F, W, time_used] = isc_cpu(Y, F, gamma, lambda, opts)
% Integrated Subspace Clustering (ISC) Main Function
% Inputs:
%   Y - Data matrix (d x N)
%   F - Initial cluster indicator matrix
%   W - Initial weight matrix
%   options - Structure containing hyperparameters (lambda, beta, gamma, alpha, etc.)

tic; 
maxIter = 50;
beta = 1e-3;
alpha = gamma;

d = opts.d;
N = opts.n;
W = ones(d,N); 

% Setting penalty parameters for the ADMM
mu1 = lambda * 1/computeLambda_mat(Y);
mu2 = lambda * 1;

C1 = zeros(N, N);
Lambda2 = zeros(N, N);
Lambda3 = zeros(1, N);
i = 1;

% Normalize F and compute initial distance matrix Q
F_norm = F * diag(sqrt(1./diag(F'*F))); 
Q = L2_distance_1(F_norm, F_norm);

while (i < maxIter)
    YW = Y .* W;
    
    % Updating Z (Representation Matrix)
    if d < N
        epsilon = 2*mu1/mu2;
        J = eye(N) - epsilon*(YW)'/ ((eye(d) + epsilon*YW*(YW)')) * YW; 
        temp_A = sum(J, 2);
        A = J - (temp_A * temp_A') / (1 + sum(temp_A));
        Z = A * ((2*mu1*((YW)'*Y)/mu2 + (C1 - Lambda2/mu2) + ones(N,1)*(ones(1,N) - Lambda3/mu2)) - beta*Q/mu2);
    else
        Z = (2*mu1*((YW)'*(YW)) + mu2*eye(N) + mu2*ones(N,N)) \ ...
            (2*mu1*((YW)'*Y) + mu2*(C1 - Lambda2/mu2) + mu2*ones(N,1)*(ones(1,N) - Lambda3/mu2)) - beta*Q;
    end
    Z = Z - diag(diag(Z)); % Zero out diagonal

    % Updating C (Sparse representation via soft thresholding)
    C2 = max(0, (abs(Z + Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z + Lambda2/mu2);
    C2 = C2 - diag(diag(C2));

    % Updating W (Feature Weights via L-BFGS-B)
    opts.grad = 1;
    if opts.grad
        [W] = LBFGSB_W(Y, F, Z, W, lambda, gamma, alpha);
    end    

    % Updating F (Cluster Indicators)
    opts.updateY = 1;
    if opts.updateY
        D = diag(sum(C2));
        L = D - (C2 + C2')/2;
        S2 = sum(W,1)'*sum(W,1);
        L = beta * L; 
        A = gamma * S2;
        F = solve_Y(L, A, F);
        
        F_norm = F * diag(sqrt(1./diag(F'*F))); 
        Q = L2_distance_1(F_norm, F_norm);
    end

    % ADMM Dual Variable Updates
    Lambda2 = Lambda2 + mu2 * (Z - C2);
    Lambda3 = Lambda3 + mu2 * (ones(1,N)*Z - ones(1,N));  
    C1 = C2;
    i = i + 1;    
end
time_used = toc;
end

%% --- Subfunctions ---

function lambda = computeLambda_mat(Y, P)
% Calculate the regularization parameter lambda based on data variance
if (nargin < 2), P = Y; end
N = size(Y, 2);
T = P' * Y;
T(1:N, :) = T(1:N, :) - diag(diag(T(1:N, :)));
T = abs(T);
lambda = min(max(T, [], 1));
end

function [W_result] = LBFGSB_W(X, F, S, W_init, lambda, gamma, alpha)
% Wrapper for L-BFGS-B optimization to solve for W
[d, n] = size(W_init);
XXS = 2 .* X .* (X * S');
SS = S * S';
YY = F' * F;
funObj = @(x) ISC_obj_W_12(X, F, S, XXS, SS, YY, lambda, alpha, gamma, n, d, x);

x_init = reshape(W_init, d*n, 1);
LB = zeros(d*n, 1);
UB = ones(d*n, 1);

opts = [];
opts.x0 = x_init;
opts.pgtol = 1e-3;
opts.maxIts = 1;
opts.printEvery = -1; % Suppress output

x = ISC_lbfgsb(funObj, LB, UB, opts);
W_result = reshape(x, d, n);
end

function [obj, grad] = ISC_obj_W_12(X, F, S, XXS, SS, YY, lambda, alpha, gamma, n, d, x)
% Objective function and gradient for W optimization
W = reshape(x, d, n);

% Terms calculation
term1 = X - (X .* W) * S;
obj_term1 = lambda * sum(sum(term1.^2));
term2 = sum(W, 1);
obj_term2 = alpha * sum(term2.^2);
term3 = sum(W * F', 1);
obj_term3 = gamma * sum(term3.^2);

obj = obj_term1 + obj_term2 + obj_term3;

% Gradient calculation
w_tmp = sum(W, 1);
temp = repmat(w_tmp, d, 1);
grad1 = lambda .* 2 .* X .* ((X .* W) * SS) - XXS;    
grad2 = 2 .* alpha * temp;
w_tmp2 = w_tmp * YY;
grad3 = 2 .* gamma * repmat(w_tmp2, d, 1);

grad = grad1 + grad2 + grad3;
grad = reshape(grad, d*n, 1);
end

function [F, obj] = solve_Y(K, A, F)
% Discrete optimization for Cluster Indicator Matrix F
fKf = sum(F .* (K * F')')';
ff = sum(F, 1)';
fAf = sum(F .* (A * F')')';

% Replace vec2ind with equivalent logic to remove Deep Learning Toolbox dependency
[~, m_all] = max(F, [], 2);

iter = 1;
for j = 1:iter
    for i = 1:size(F, 1)
        m = m_all(i);
        if ff(m) == 1, continue; end % Avoid empty clusters
    
        Y_F = F' * K(:, i);
        Y_A = F' * A(:, i);

        % Gains/Losses when moving i to other clusters
        fKf_s = fKf + 2 * Y_F + K(i, i); fKf_s(m) = fKf(m);
        ff_k = ff + 1; ff_k(m) = ff(m);
        fAf_s = fAf + 2 * Y_A + A(i, i); fAf_s(m) = fAf(m);
    
        fKf_0 = fKf; fKf_0(m) = fKf(m) - 2 * Y_F(m) + K(i, i);
        ff_0 = ff; ff_0(m) = ff(m) - 1;
        fAf_0 = fAf; fAf_0(m) = fAf(m) - 2 * Y_A(m) + A(i, i);
    
        delta = fKf_s ./ ff_k - fKf_0 ./ ff_0 + fAf_s - fAf_0;
        [~, p] = min(delta);
        
        if p ~= m
            fKf([m, p]) = [fKf_0(m), fKf_s(p)];
            fAf([m, p]) = [fAf_0(m), fAf_s(p)];
            ff([m, p]) = [ff_0(m), ff_k(p)];
            F(i, [p, m]) = [1, 0];
            m_all(i) = p;
        end
    end
    obj = sum(fKf ./ ff);
end    
end

function D = L2_distance_1(a, b)
% Compute squared Euclidean distance between columns
if nargin < 2, b = a; end
aa = sum(a.*a, 1); bb = sum(b.*b, 1);
ab = a' * b;
D = bsxfun(@plus, aa', bb) - 2 * ab;
D(D < 0) = 0;
end


function val = h_getOpt(opts, field, default)
if isfield(opts, field), val = opts.(field); else val = default; end
end