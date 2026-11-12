function [F_star, G_star, time_used] = mlsc_cpu(X, alpha, beta, opts)
    % MLSC: Metric Learning-Based Subspace Clustering
    %
    % Reference: 
    % Xu et al., "Metric Learning-Based Subspace Clustering", IEEE TNNLS 2025.
    %
    % Inputs:
    %   X       - Data matrix (d x n), d is feature dim, n is number of samples
    %   c       - Number of clusters
    %   alpha   - Hyper-parameter for adaptive neighbors learning
    %   beta    - Hyper-parameter for self-expressive regularization
    %   options - Struct for algorithmic parameters (optional)
    %             .h    : Dimension of latent space for P (default 50)
    %             .eta  : Learning rate for P (default 1e-5)
    %             .v    : Growth factor for mu (default 1.2)
    %             .mu   : Initial penalty parameter (default 1e-6)
    %             .max_mu: Maximum value for mu (default 1e8)
    %             .eps  : Convergence tolerance (default 1e-8)
    %             .max_iter: Maximum ADMM iterations (default 100)
    %
    % Outputs:
    %   F_star  - Learned similarity matrix (n x n)
    %   G_star  - Learned pairwise graph matrix (n x n)
    %   Y       - Discrete cluster labels (n x 1)

    % --- Set default parameters ---
    tic
    if ~isfield(opts, 'h'); opts.h = 50; end
    if ~isfield(opts, 'eta'); opts.eta = 1e-5; end
    if ~isfield(opts, 'v'); opts.v = 1.2; end
    if ~isfield(opts, 'mu'); opts.mu = 1e-6; end
    if ~isfield(opts, 'max_mu'); opts.max_mu = 1e8; end
    if ~isfield(opts, 'eps'); opts.eps = 1e-8; end
    if ~isfield(opts, 'max_iter'); opts.max_iter = 100; end

    d = opts.d;
    n = opts.n;

    eta = opts.eta;
    v = opts.v;
    mu = opts.mu;
    max_mu = opts.max_mu;
    tol = opts.eps;
    max_iter = opts.max_iter;

    % Mean-centered original data (x_i - \hat{x}_i)
    X_cen = X - mean(X, 1); 

    % --- Initialization ---
    G = zeros(n, n);
    H = zeros(n, n);
    F = zeros(n, n);
    W1 = zeros(n, n);
    W2 = zeros(n, n);
    
    % Randomly initialize learnable metric parameter P (h x d)
    rng(42); % For reproducibility
    P = randn(opts.h, d) * 0.01;

    I_n = eye(n);

    for iter = 1:max_iter
        % ---------------------------------------------------------
        % Step 1: Update P (Linearity-aware distance metric)
        % Using analytically derived gradient for Eq. (9)
        % ---------------------------------------------------------
        U = P * X_cen; % h x n
        norms_U = sqrt(sum(U.^2, 1)) + 1e-8; % 1 x n, add eps to prevent div by 0
        U_hat = bsxfun(@rdivide, U, norms_U); % Normalized projections
        
        % Compute Linearity-aware Distance Matrix
        S_cos = U_hat' * U_hat; % Cosine similarity
        Dist2 = 1 - S_cos;      % Squared linearity-aware distance
        
        % Vectorized analytical gradient computation for \nabla_P
        V = U_hat * G; 
        d_vec = sum(U_hat .* V, 1); % Diagonal of U_hat' * U_hat * G
        
        % grad_U calculation derived using matrix calculus
        grad_U = -2 * bsxfun(@rdivide, V - bsxfun(@times, U_hat, d_vec), norms_U);
        grad_P = grad_U * X_cen';
        
        % Gradient descent step
        P = P - eta * grad_P;

        % ---------------------------------------------------------
        % Step 2: Update F (Self-expressiveness representation)
        % Solving Eq. (10) explicitly: F = (\mu H'H + 2\beta I)^{-1} (\mu H'H + H'W1)
        % ---------------------------------------------------------
        HtH = H' * H;
        F = 1/(2 * beta )*(mu * HtH +  I_n) \ (mu * HtH + H' * W1);

        % ---------------------------------------------------------
        % Step 3: Update H
        % Solving Eq. (12) explicitly: H * B = A -> H = A / B
        % ---------------------------------------------------------
        I_minus_F = I_n - F;
        A_H = mu * G + W2 - W1 * I_minus_F';
        B_H = 2 * alpha * I_n + mu * (I_minus_F * I_minus_F') + mu * I_n;
        H = A_H / B_H;

        % ---------------------------------------------------------
        % Step 4: Update G (Adaptive Pairwise Graph)
        % Solving Eq. (15) & (8) utilizing strict Simplex Projection
        % ---------------------------------------------------------
        % K_ij = Dist_P^2(x_i, x_j) - \mu h_ij + w^{(2)}_ij
        K_mat = Dist2 - mu * H + W2;
        K_tilde = - (1 / mu) * K_mat;
        
        for i = 1:n
            G(i, :) = project_to_simplex(K_tilde(i, :));
        end

        % ---------------------------------------------------------
        % Step 5: Update Multipliers and Penalty Parameter
        % ---------------------------------------------------------
        W1 = W1 + mu * (H - H * F);
        W2 = W2 + mu * (G - H);
        mu = min(v * mu, max_mu);

        % ---------------------------------------------------------
        % Step 6: Convergence Check
        % ---------------------------------------------------------
        err1 = norm(H - H * F, 'fro')^2;
        err2 = norm(G - H, 'fro')^2;
        
        if err1 < tol && err2 < tol
            time_used = toc;
            break;
        end
    end

    F_star = F;
    G_star = G;
    if iter >= max_iter
        time_used = toc;
    end

end

% -------------------------------------------------------------------------
% Helper Function: Exact Simplex Projection (Satisfies Eq. 22-24 exactly)
% -------------------------------------------------------------------------
function x = project_to_simplex(v)
    % Projects vector v onto the probability simplex: sum(x)=1, x>=0
    n = length(v);
    u = sort(v, 'descend');
    sum_u = 0;
    rho = 0;
    for j = 1:n
        sum_u = sum_u + u(j);
        if u(j) + (1 - sum_u) / j > 0
            rho = j;
        else
            break;
        end
    end
    theta = (1 - sum(u(1:rho))) / rho;
    x = max(v + theta, 0);
end

% -------------------------------------------------------------------------
% Helper Function: Standard Spectral Clustering
% -------------------------------------------------------------------------
function Y = spectral_clustering(S, c)
    % Compute the degree matrix and normalized Laplacian
    D = diag(sum(S, 2));
    D_inv_sqrt = diag(1 ./ (sqrt(diag(D)) + eps));
    L_norm = eye(size(S, 1)) - D_inv_sqrt * S * D_inv_sqrt;
    
    % Ensure numerical symmetry
    L_norm = (L_norm + L_norm') / 2;
    
    % Eigen decomposition
    [V, ~] = eigs(L_norm, c, 'smallestabs');
    
    % Normalize rows to unit length for clustering
    V_norm = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)) + eps);
    
    % K-means to get final discrete labels
    Y = kmeans(V_norm, c, 'Replicates', 20, 'EmptyAction', 'singleton');
end