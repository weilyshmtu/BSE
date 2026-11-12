function X_proj = project_to_simplex_matrix_cpu(X)
    % Vectorized Simplex Projection for all columns simultaneously
    [d, n] = size(X);
    
    % Step 1: Sort all columns in descending order
    Xs = sort(X, 1, 'descend');
    Xsum = cumsum(Xs, 1);
    
    % Step 2: Find the active set threshold (rho) for each column
    idx = (1:d)';
    cond = (Xs - (Xsum - 1) ./ idx) > 0;
    rho_idx = sum(cond, 1); 
    
    % Step 3: Compute theta using linear indices 
    lin_idx = rho_idx + ((1:n) - 1) * d;
    theta = (Xsum(lin_idx) - 1) ./ rho_idx;
    
    % Step 4: Truncate and Output
    X_proj = max(X - theta, 0);
    
    % Numerical stability: Strictly enforce sum to 1
    X_proj = X_proj ./ (sum(X_proj, 1) + eps);
end