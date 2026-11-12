function groups = SpectralClustering_Fast(CKSym, n)
    N = size(CKSym, 1);
    
    % --- Fix 1: Force S to be strictly symmetric ---
    % This is critical because numerical noise in the previous 
    % Alternating Descent steps breaks perfect symmetry.
    CKSym = (CKSym + CKSym') / 2;
    
    if ~issparse(CKSym), CKSym = sparse(CKSym); end
    
    d = sum(CKSym, 2);
    d_inv_sqrt = 1 ./ sqrt(d + eps);
    DN = spdiags(d_inv_sqrt, 0, N, N);
    
    % L = I - D^(-1/2) * C * D^(-1/2)
    L = speye(N) - DN * CKSym * DN;
    
    % Ensure L is strictly symmetric before eigendecomposition
    L = (L + L') / 2;
    
    % --- Fix 2: Use 'sr' (Smallest Real) to hint eigs to look for real eigenvalues ---
    % eigs with 'smallestabs' can sometimes return complex conjugate pairs
    % if the matrix is slightly non-symmetric due to noise.
    [kerN, ~] = eigs(L, n, 'sr');
    
    % --- Fix 3: Force real numbers and handle potential imaginary noise ---
    % Even if eigs returns a complex matrix, the imaginary parts 
    % should be negligible (near 0). We discard them explicitly.
    kerN = real(kerN);
    
    % 4. Vectorized row-wise normalization
    rowNorms = sqrt(sum(kerN.^2, 2)) + eps;
    kerNS = kerN ./ rowNorms;
    
    % 5. k-means with replicates
    groups = kmeans(kerNS, n, 'MaxIter', 1000, 'Replicates', 5, 'Start', 'plus');
end