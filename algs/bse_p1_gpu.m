function [C, S, info] = bse_p1_gpu(X, alpha, opts)
%==========================================================================
% P1 (GPU): MM solver for the squared-difference co-regularized model.
%
% Model (identical to the paper):
%   min  F(C,S) = ||X - X*C||_F^2 + <P, S>
%        + alpha * ||C-S||_F^2
%   s.t. C,S >= 0, each column sums to 1, zero diagonal.
%
% Updates per outer iteration:
%   C update: MM quadratic upper bound + exact weighted simplex projection;
%   S update: exact constrained block minimization (weighted QP on the
%             simplex), using the new C (Gauss-Seidel).
%
% Implementation notes (sources of speed):
%   P1 ablation: R(C,S) = ||C-S||_F^2.
%   (a) full GPU residency: no CPU<->GPU transfer inside the main loop;
%   (b) K = X'X is formed once to compute a rigorous rho; the main-loop
%       gradient uses X*C and X'*(X*C-X), i.e., 2*d*n^2 flops per iteration;
%   (c) exact weighted simplex projection by one sort + cumulative sums.
%
% Memory: full GPU residency needs about 8 dense n-by-n matrices;
%   12 GB of GPU memory supports n up to roughly 18000 in single precision.
%
% opts fields:
%   Dx, S_init         required; C_init is optional (defaults to S_init)
%   n                  optional consistency check; defaults to size(X,2)
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

    if ~isscalar(alpha) || ~isfinite(alpha) || alpha <= 0
        error('alpha must be a positive finite scalar.');
    end
    n = size(X, 2);
    if isfield(opts, 'n') && ~isempty(opts.n) && opts.n ~= n
        error('opts.n must equal size(X, 2).');
    end
    maxIter   = get_opt(opts, 'max_iter', get_opt(opts, 'maxIter', 500));
    minIter   = get_opt(opts, 'minIter',   20);
    tol       = get_opt(opts, 'tol',       1e-5);
    objEvery  = get_opt(opts, 'objEvery',  1);
    verbose   = get_opt(opts, 'verbose',   true);
    prec      = get_opt(opts, 'precision', 'single');
    % --- Move everything to the GPU (single precision by default) ---
    Xg = full(gpuArray(cast(X, prec)));
    P  = full(gpuArray(cast(opts.Dx, prec)));
    S  = full(gpuArray(cast(opts.S_init, prec)));
    if isfield(opts, 'C_init') && ~isempty(opts.C_init)
        C = full(gpuArray(cast(opts.C_init, prec)));
    else
        C = S;
    end

    alpha = cast(alpha, 'like', Xg);
    diagIdx = 1:(n+1):(n*n);

    % Project the initializers onto the (zero-diagonal) simplex
    C = weighted_simplex_projection(C, ones(size(C), 'like', Xg), diagIdx);
    S = weighted_simplex_projection(S, ones(size(S), 'like', Xg), diagIdx);

    % Rigorous MM constant: lambda_max(K) <= min(||K||_F, ||K||_inf).
    K = Xg.' * Xg;
    rho = min(norm(K, 'fro'), norm(K, inf));
    clear K;  % K is not needed in the main loop
    if gather(rho) <= 0
        error('rho must be positive (X should not be identically zero).');
    end

    info.obj  = nan(maxIter, 1);
    info.resC = nan(maxIter, 1);
    info.resS = nan(maxIter, 1);
    info.rho  = gather(rho);
    info.iter = 0;
    info.converged = false;
    info.regularizer = '||C-S||_F^2';

    %======================================================================
    % Main loop (two d-by-n^2 GEMMs per iteration, no CPU transfers)
    %======================================================================
    for it = 1:maxIter
        C_old = C;
        S_old = S;

        % ---- 1) C update: MM upper bound + exact weighted projection ----
        XC = Xg * C;                              % d-by-n
        G  = 2 * (Xg.' * (XC - Xg));              % gradient of ||X-XC||^2
        aC = (rho + alpha) * ones(size(C_old), 'like', Xg);
        vC = (2*rho*C_old - G + 2*alpha*S_old) ./ (2*aC);
        C  = weighted_simplex_projection(vC, aC, diagIdx);

        % ---- 2) S update: exact constrained block minimization
        %         (Gauss-Seidel, uses the new C) ----
        aS = alpha * ones(size(C), 'like', Xg);
        vS = C - P ./ (2*alpha);
        S  = weighted_simplex_projection(vS, aS, diagIdx);

        % Relative iterate changes (two gathers per iteration)
        resC = sqrt(sum((C - C_old).^2, 'all')) / max(1, sqrt(sum(C.^2, 'all')));
        resS = sqrt(sum((S - S_old).^2, 'all')) / max(1, sqrt(sum(S.^2, 'all')));
        resC_g = gather(resC);  resS_g = gather(resS);
        info.resC(it) = resC_g;
        info.resS(it) = resS_g;

        % Objective value, recomputed with the current (C,S)
        if mod(it, objEvery) == 0 || it == 1
            info.obj(it) = gather(p1_objective(Xg, P, C, S, alpha));
        end

        if verbose && (mod(it, 100) == 0 || it == 1)
            fprintf('P1-MM | iter %4d | obj %.6e | resC %.3e | resS %.3e\n', ...
                    it, info.obj(it), resC_g, resS_g);
        end

        if max(resC_g, resS_g) < tol && it >= minIter
            info.iter = it;
            info.converged = true;
            break;
        end
    end

    if info.iter == 0
        info.iter = maxIter;
        % Record the final objective value as well
        info.obj(maxIter) = gather(p1_objective(Xg, P, C, S, alpha));
    end
    info.time = toc;
    info.obj  = info.obj(1:info.iter);
    info.resC = info.resC(1:info.iter);
    info.resS = info.resS(1:info.iter);

    C(diagIdx) = 0;
    S(diagIdx) = 0;
    C = gather(C);
    S = gather(S);
end

%==========================================================================
% Helper functions
%==========================================================================

function Z = weighted_simplex_projection(V, A, diagIdx)
% Exact weighted column simplex projection by one sort + cumulative sums.
% Solves min_z sum_i a_i (z_i-v_i)^2  s.t. z_i>=0, sum_i z_i=1, z_j=0.
    [n,~] = size(V);
    U = A.*V;
    U(diagIdx) = cast(-inf,'like',V);
    [Us, ord] = sort(U, 1, 'descend');
    colOff = cast((0:n-1)*n, 'like', ord);
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

function val = p1_objective(X, P, C, S, alpha)
% Full objective value (X*C is recomputed with the current C).
    rec = X * C - X;
    reg = sum((C(:) - S(:)).^2);
    val = sum(rec(:).^2) + sum(P(:) .* S(:)) + alpha * reg;
end

function v = get_opt(opts, name, defaultValue)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = defaultValue;
    end
end
