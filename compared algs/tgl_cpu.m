function [G_final,time_used] = tgl_cpu(lambda1, lambda2, opts)
    % X: d x n data matrix (d: dimensionality, n: samples)
    % lambda1, lambda2: balancing parameters from the paper
    tic
    n = opts.n;
    Dx = opts.Dx;
    XtX = opts.XtX;
    
    %  Initialize variables
    Gn = zeros(n, n);
    Gs = zeros(n, n);
    I = eye(n); P = I;
    C = zeros(n, 2, n); % Tensor size n x 2 x n as defined in Section III
    Y = zeros(n, n);    % Multiplier for Gs - P = 0
    W = zeros(n, 2, n); % Multiplier for G - C = 0
    mu = 1e-4; rho = 1.1; max_mu = 1e6;
    maxIter = 100;
    tol = 1e-6;
    inv_part = 2 * lambda2 * XtX;
    
    for iter = 1:maxIter
        % --- Update Gn (Neighbor Graph via Simplex Projection) ---
        Qn = squeeze(C(:, 1, :)) - (1/mu) * squeeze(W(:, 1, :)) - (lambda1/mu) * Dx;
        Gn = EProjSimplex_new(Qn); 
        
        % --- Update Gs (Self-expressiveness Graph) ---
        Qs = 0.5 * (P + squeeze(C(:, 2, :))) - (1/(2*mu)) * (Y + squeeze(W(:, 2, :)));
        Gs = EProjSimplex_new(Qs);
        
        % --- Update P (Auxiliary variable for self-representation) ---
        P = (inv_part + mu * I) \ (inv_part + Y + mu * Gs);
        
        % --- Update Tensor C (via Tensor SVT in FFT domain) ---
        G_tensor(:, 1, :) = Gn;
        G_tensor(:, 2, :) = Gs;
        Temp = G_tensor + (1/mu) * W;
        C = tensor_svt(Temp, 1/mu);
        
        % --- Update Multipliers and Penalty Parameter ---
        Y = Y + mu * (Gs - P);
        W = W + mu * (G_tensor - C);
        mu = min(mu * rho, max_mu);
        
        % Check convergence
        err = max(norm(Gs - P, 'fro')/n, norm(G_tensor(:) - C(:), 'fro')/n);
        if err < tol 
            time_used = toc;
            break; 
        end
    end
    
    % Final Graph Fusion (Eq. 7)
    G_final = 0.25 * (Gn + Gn' + Gs + Gs');
    if iter == maxIter
        time_used = toc;
    end
end


% Helper: Tensor Singular Value Thresholding
function C = tensor_svt(T, kappa)
    [n1, n2, n3] = size(T);
    T_f = fft(T, [], 2); % FFT along the 3rd dimension
    C_f = zeros(n1, n2, n3);
    for i = 1:n2
        [U, S, V] = svd(squeeze(T_f(:, i, :)), 'econ'); % SVD on each frontal slice
        S = diag(S);
        S = max(S - kappa, 0);
        C_f(:, i, :) = U * diag(S) * V';
    end
    C = real(ifft(C_f, [], 2));
end

% Helper: Column-wise Simplex Projection
function S = EProjSimplex_new(V)
    [n, m] = size(V);
    V_sorted = sort(V, 1, 'descend');
    V_cum = cumsum(V_sorted, 1);
    cond = V_sorted - (V_cum - 1) ./ (1:n)' > 0;
    [~, rho] = max(cond .* (1:n)', [], 1);
    theta = (V_cum(sub2ind([n, m], rho, 1:m)) - 1) ./ rho;
    S = max(V - theta, 0);
end