
function [S] = project_to_simplex_blk(opts.S_init)
blkSize = 2000; 
% --- Step 1: Chunked Initialization of Matrix S ---
S = gpuArray.zeros(n, n);
for i = 1:blkSize:n
    en = min(i + blkSize - 1, n);
    cols_count = en - i + 1;
    
    % Load a vertical slice of S_init to VRAM
    S_init_sub = gpuArray(opts.S_init(:, i:en));
    
    % Calculate local linear indices for the global diagonal elements
    j_idx = i:en;
    sub_diag_idx = j_idx + (j_idx - i) * n;
    S_init_sub(sub_diag_idx) = -inf; 
    
    % Project chunk to simplex and store
    S(:, i:en) = project_to_simplex_chunk(S_init_sub, d_vec);
    clear S_init_sub;
end
end

function X_proj_sub = project_to_simplex_chunk(X_sub, d_vec)
    % Vectorized Simplex Projection executed over a chunk of columns
    X_sub(isnan(X_sub)) = -inf;
    [~, cols_count] = size(X_sub);
    
    Xs = sort(X_sub, 1, 'descend');
    Xsum = cumsum(Xs, 1);
    
    cond = (Xs - (Xsum - 1) ./ d_vec) > 0;
    rho_idx = sum(cond, 1); 
    rho_idx(rho_idx < 1) = 1; 
    
    % Vectorized lookup using chunk-relative linear offsets
    sub_offsets = gpuArray(0:(cols_count-1)) * size(X_sub, 1);
    lin_idx = cast(rho_idx, 'int64') + cast(sub_offsets, 'int64');
    
    theta = (Xsum(lin_idx) - 1) ./ rho_idx;
    X_proj_sub = max(X_sub - theta, 0);
    % Added 'max(..., eps)' to avoid zero-division in normalization step
    X_proj_sub = X_proj_sub ./ max(sum(X_proj_sub, 1), eps);
end
