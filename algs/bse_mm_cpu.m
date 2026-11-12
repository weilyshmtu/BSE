function [C, S, info] = bse_mm_cpu(X, alpha, opts)
%==========================================================================
% BSE-MM (CPU): monotone MM solver for the BSE model.
%
% CPU counterpart of bse_mm_gpu with identical mathematics and
% defaults; intended for CPU-vs-GPU timing comparisons and for machines
% without a GPU.
%
% Model (identical to the paper):
%   min  F(C,S) = ||X - X*C||_F^2 + <P, S>
%        + alpha * sum( C^2.*(t-S)^2 + S^2.*(t-C)^2 )
%   s.t. C,S >= 0, each column sums to 1, zero diagonal, t = 1/k.
%
% Updates per outer iteration:
%   C update: MM quadratic upper bound + exact weighted simplex projection;
%   S update: exact constrained block minimization (weighted QP on the
%             simplex), using the new C (Gauss-Seidel).
%
% Implementation notes:
%   (a) K = X'X is formed once to compute a rigorous rho; the main-loop
%       gradient uses X*C and X'*(X*C-X), i.e., 2*d*n^2 flops per iteration;
%   (b) exact weighted simplex projection by one sort + cumulative sums.
%
% For a fair CPU-vs-GPU timing comparison:
%   - use the SAME precision in both versions ('single' or 'double');
%   - MATLAB's CPU BLAS already uses all cores by default; control the
%     thread count with maxNumCompThreads if a specific setting is
%     reported in the paper.
%
% opts fields:
%   k, Dx, S_init   required; n defaults to size(X,2);
%                   C_init optional (defaults to S_init)
%   max_iter  : outer iteration budget, default 500
%   minIter   : minimum outer iterations before stopping, default 20
%   tol       : relative iterate-change threshold, default 1e-5
%   objEvery  : objective logging interval, default 1
%   (rho is computed internally as min{||X'X||_F, ||X'X||_inf};
%    no rho-related option is required.)
%   precision : 'single' (default) or 'double'
%   verbose   : default true
%==========================================================================
    tic
    if nargin < 3 || isempty(opts), opts = struct(); end

    k         = opts.k;
    n         = opts.n;
    maxIter   = get_opt(opts, 'max_iter',   500);
    minIter   = get_opt(opts, 'minIter',   20);
    tol       = get_opt(opts, 'tol',       1e-5);
    objEvery  = get_opt(opts, 'objEvery',  1);
    verbose   = get_opt(opts, 'verbose',   true);
    prec      = get_opt(opts, 'precision', 'single');
    Xc = cast(X, prec);
    P  = cast(opts.Dx, prec);
    S  = cast(opts.S_init, prec);
    if isfield(opts, 'C_init') && ~isempty(opts.C_init)
        C = cast(opts.C_init, prec);
    else
        C = S;
    end

    alpha = cast(alpha, 'like', Xc);
    t     = cast(1 / k, 'like', Xc);
    diagIdx = 1:(n+1):(n*n);

    % Project the initializers onto the (zero-diagonal) simplex
    C = weighted_simplex_projection(C, ones(size(C), 'like', Xc), diagIdx);
    S = weighted_simplex_projection(S, ones(size(S), 'like', Xc), diagIdx);

    % Rigorous MM constant: lambda_max(K) <= min(||K||_F, ||K||_inf).
    K = opts.XtX;
    rho = min(norm(K, 'fro'), norm(K, inf));
    clear K;  % K is not needed in the main loop
    if rho <= 0
        error('rho must be positive (X should not be identically zero).');
    end

    info.obj  = nan(maxIter, 1);
    info.resC = nan(maxIter, 1);
    info.resS = nan(maxIter, 1);
    info.rho  = rho;
    info.iter = 0;
    info.converged = false;
    info.device = 'cpu';

    %======================================================================
    % Main loop (two d-by-n^2 GEMMs per iteration)
    %======================================================================
    for it = 1:maxIter
        C_old = C;
        S_old = S;

        % ---- 1) C update: MM upper bound + exact weighted projection ----
        XC = Xc * C;                              % d-by-n
        G  = 2 * (Xc.' * (XC - Xc));              % gradient of ||X-XC||^2
        aC = rho + alpha * ((t - S_old).^2 + S_old.^2);
        vC = (2*rho*C_old - G + 2*alpha*t*(S_old.^2)) ./ (2*aC);
        C  = weighted_simplex_projection(vC, aC, diagIdx);

        % ---- 2) S update: exact constrained block minimization
        %         (Gauss-Seidel, uses the new C) ----
        aS = alpha * (C.^2 + (t - C).^2);
        vS = (2*alpha*t*(C.^2) - P) ./ (2*aS);
        S  = weighted_simplex_projection(vS, aS, diagIdx);

        % Relative iterate changes
        resC = sqrt(sum((C - C_old).^2, 'all')) / max(1, sqrt(sum(C.^2, 'all')));
        resS = sqrt(sum((S - S_old).^2, 'all')) / max(1, sqrt(sum(S.^2, 'all')));
        info.resC(it) = resC;
        info.resS(it) = resS;

        % Objective value, recomputed with the current (C,S)
        if mod(it, objEvery) == 0 || it == 1
            info.obj(it) = bse_objective(Xc, P, C, S, alpha, t);
        end

        if verbose && (mod(it, 100) == 0 || it == 1)
            fprintf('BSE-MM CPU | iter %4d | obj %.6e | resC %.3e | resS %.3e\n', ...
                    it, info.obj(it), resC, resS);
        end

        if max(resC, resS) < tol && it >= minIter
            info.iter = it;
            info.converged = true;
            break;
        end
    end

    if info.iter == 0
        info.iter = maxIter;
        % Record the final objective value as well
        info.obj(maxIter) = bse_objective(Xc, P, C, S, alpha, t);
    end
    info.time = toc;
    info.obj  = info.obj(1:info.iter);
    info.resC = info.resC(1:info.iter);
    info.resS = info.resS(1:info.iter);

    C(diagIdx) = 0;
    S(diagIdx) = 0;
end

%==========================================================================
% Helper functions
%==========================================================================

function Z = weighted_simplex_projection(V, A, diagIdx)
% Exact weighted column simplex projection by one sort + cumulative sums.
% Solves min_z sum_i a_i (z_i-v_i)^2  s.t. z_i>=0, sum_i z_i=1, z_j=0.
    [n,~] = size(V);
    U = A.*V;
    U(diagIdx) = -inf;
    [Us, ord] = sort(U, 1, 'descend');
    colOff = (0:n-1)*n;
    lin = ord + colOff;
    invA = 1./A;
    Vs  = V(lin);
    Gs  = invA(lin);
    csV = cumsum(Vs,1);
    csG = cumsum(Gs,1);
    tauM = (csV - 1)./csG;
    valid = Us > tauM;
    valid(1,:) = true;
    mc = sum(valid,1);
    tau = tauM(mc + (0:n-1)*n);
    Z = max(0, V - tau./A);
    Z(diagIdx) = 0;
end

function val = bse_objective(X, P, C, S, alpha, t)
% Full objective value (X*C is recomputed with the current C).
    rec = X * C - X;
    reg = sum(C(:).^2 .* (t - S(:)).^2 + S(:).^2 .* (t - C(:)).^2);
    val = sum(rec(:).^2) + sum(P(:) .* S(:)) + alpha * reg;
end

function v = get_opt(opts, name, defaultValue)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = defaultValue;
    end
end
