function [F, S, time_used] = dsc_cpu(A, k, opts)
    % DSC: Direct Spectral Clustering
    %
    % Reference:
    % Feiping Nie et al., "A Novel and Effective Method to Directly Solve
    % Spectral Clustering", IEEE TPAMI 2024.
    %
    % Inputs:
    %   A       - Initial similarity matrix (n x n)
    %   c       - Number of clusters
    %   k       - Number of nearest neighbors (for adaptive alpha computation)
    %   options - Struct containing hyperparameters:
    %             .lambda   : Initial tradeoff parameter (default 10)
    %             .max_iter : Maximum number of iterations (default 30)
    %             .t        : Interval bound for relaxed matrix S [-t, t] (default 1.0)
    %
    % Outputs:
    %   F       - Weighted discrete indicator matrix (n x c)
    %   S       - Learned structured similarity matrix (n x n)
    %   Y       - Discrete cluster labels (n x 1)

    tic
    max_iter = 30;
    lambda = 10;
    tol = 1e-10;

    n = opts.n;
    c = opts.n_clusters;
    % --- Step 1: Initialization ---
    % Compute the Graph Laplacian L_A
    D_A = diag(sum(A, 2));
    L_A = D_A - (A + A') / 2;

    % Initialize F using the c eigenvectors of L_A corresponding to c smallest eigenvalues
    % (Using 'smallestabs' to compute efficiently for large sparse matrices)
    [F, ~] = eig1(L_A, c, 0);

    for iter = 1:max_iter
        % --- Step 2: Update Similarity Matrix S ---
        % Compute squared Euclidean distance matrix between rows of F
        D_dist = pdist2(F, F, 'squaredeuclidean');

        % Compute adaptive alpha (Based on Eq. 39 & 41)
        dS = sort(D_dist, 2, 'ascend');
        % The k nearest neighbors are indices 2 to k+1 (index 1 is self-distance 0)
        alpha_i = (lambda / 4) * k * dS(:, k+2) - (lambda / 4) * sum(dS(:, 2:k+1), 2);
        alpha = mean(alpha_i);

        S = zeros(n, n);
        for i = 1:n
            u = - (lambda / (4 * alpha)) * D_dist(i, :);
            
            S(i, :) = project_to_simplex(u);
        end

        S = S - diag(diag(S)); % Remove self-loops
        S = (S + S') / 2;      % Force strict symmetry

        % --- Step 3: Update Indicator Matrix F ---
        % Compute Laplacian L_S. 
        % We use abs(S) for the degree matrix to handle negative similarities from the relaxed bound.
        D_S = diag(sum(abs(S), 2));
        L_S = D_S - S;

        % Construct L_tilde and ensure numerical symmetry
        L_tilde = L_A + lambda * L_S;
        L_tilde = (L_tilde + L_tilde') / 2;

        % Update F
        [F, ~, ev] = eig1(L_tilde, c, 0);

        % --- Step 4: Dynamic Lambda Adjustment (Section V.E.2) ---
        % Compute the number of connected components (zero eigenvalues of L_S)
        F_old = F;
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1)); 
        flag = sum(logical(diag(D_S-eye(n))<0.01));
        if fn1 > tol 
            lambda = 2*lambda;
        elseif fn2 < tol
            lambda = lambda/2;  
            F = F_old;
        elseif fn1 < tol && fn2 > tol && flag == n  % output graph in advance [when the graph is structured and doubly stochastic] 
            time_used = toc;
            break;
        end
        if iter == max_iter
            time_used = toc;
        end
    end
end

function x = project_to_simplex(v)
    n = length(v);
    u = sort(v, 'descend');
    rho = 0;
    sum_u = 0;
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