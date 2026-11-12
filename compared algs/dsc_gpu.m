function [F, S, time_used] = dsc_gpu(A, k, opts)
    tic
    % --------------------------
    % Step 1: Initial Settings
    % --------------------------
    A = gpuArray(A);
    n = opts.n;
    c = opts.n_clusters;
    max_iter = 30;
    lambda = gpuArray(10.0);
    tol = gpuArray(1e-10);
    
    % Pre-compute constant indices for Simplex
    idx_vec = gpuArray((1:n)');
    col_offsets = gpuArray(0:n-1) * n;

    D_A = diag(sum(A, 2));
    L_A = D_A - (A + A') / 2;

    % Initial F using GPU EIG
    [F, ~] = eig1_gpu(L_A, c);

    for iter = 1:max_iter
        % Step 2: Update S (Vectorized Distance Calculation)
        % ||a-b||^2 = ||a||^2 + ||b||^2 - 2a'b
        G = sum(F.^2, 2);
        D_dist = max(0, G + G' - 2*(F * F')); 

        % Adaptive alpha
        dS = sort(D_dist, 2, 'ascend');
        alpha_i = (lambda / 4) * k * dS(:, k+2) - (lambda / 4) * sum(dS(:, 2:k+1), 2);
        alpha = mean(alpha_i);

        % Vectorized project_to_simplex
        U = - (lambda / (4 * alpha)) * D_dist;
        S = project_to_simplex_matrix_gpu(U, idx_vec, col_offsets);

        S(1:n+1:end) = 0; % Remove self-loops
        S = (S + S') / 2;

        % Step 3: Update F
        D_S = diag(sum(abs(S), 2));
        L_S = D_S - S;
        L_tilde = L_A + lambda * L_S;
        L_tilde = (L_tilde + L_tilde') / 2;

        [F, ev] = eig1_gpu(L_tilde, c + 1); % Get c+1 to check fn2

        % Step 4: Dynamic Lambda
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1)); 
        % flag check (simplified for GPU)
        flag_val = sum(diag(D_S) < 1.01); 
        
        if fn1 > tol 
            lambda = 2 * lambda;
        elseif fn2 < tol
            lambda = lambda / 2; 
             F = F_old;
        elseif fn1 < tol && fn2 > tol && flag_val == n
            break;
        end
    end
    F = gather(F);
    S = gather(S);
    time_used = toc;
end

function [V, ev] = eig1_gpu(L, c)
    [V, D] = eig(L);
    [ev, idx] = sort(diag(D), 'ascend');
    V = V(:, idx(1:c));
    ev = ev(1:c);
end

function X_proj = project_to_simplex_matrix_gpu(X, idx, offsets)
    [d, n] = size(X);
    Xs = sort(X, 1, 'descend');
    Xsum = cumsum(Xs, 1);
    cond = (Xs - (Xsum - 1) ./ idx) > 0;
    rho_idx = sum(cond, 1); 
    lin_idx = rho_idx + offsets;
    theta = (Xsum(lin_idx) - 1) ./ rho_idx;
    X_proj = max(X - theta, 0);
end