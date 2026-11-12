function [C, S, info] = bse_p2_gpu(X, alpha, opts)
%==========================================================================
% BSE-MM (GPU): mixed-precision stabilized single-polarization solver.
%
% Model (identical to the paper):
%   min  F(C,S) = ||X - X*C||_F^2 + <P, S>
%        + alpha * sum( C^2.*(t-S)^2 )
%   s.t. C,S >= 0, each column sums to 1, zero diagonal, t = 1/k.
%
% Updates per outer iteration:
%   C update: MM quadratic upper bound + exact weighted simplex projection;
%   S update: exact constrained block minimization (weighted QP on the
%             simplex), using the new C (Gauss-Seidel).
%
% Implementation notes (sources of speed):
%   Single-polarization ablation: \mathcal{R}_{C\leftarrow S}=\sum_{i,j}C_{ij}^2(t-S_{ij})^2
%   (a) full GPU residency: no CPU<->GPU transfer inside the main loop;
%   (b) K = X'X is formed once to compute a rigorous rho; the main-loop
%       gradient uses X*C and X'*(X*C-X), i.e., 2*d*n^2 flops per iteration;
%   (c) sensitive simplex subproblems use internal double precision and
%       return to the outer precision; no weight clipping or relaxation.
%
% Memory: main arrays remain in the outer precision, but the stabilized
%   projections require additional double-precision temporary storage.
%
% opts fields:
%   k, n, Dx, S_init   required (C_init optional; defaults to S_init)
%   max_iter  : outer iteration budget, default 500
%   minIter   : minimum outer iterations before stopping, default 20
%   tol       : relative iterate-change threshold, default 1e-5
%   objEvery  : objective logging interval, default 1
%   (rho is computed internally as min{||X'X||_F, ||X'X||_inf};
%    no rho-related option is required.)
%   precision : 'single' (default) or 'double'. With 'single', X, C, S
%    and the main GEMMs stay single; projections/objective sums use double.
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
    t     = cast(1 / k, 'like', Xg);
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
    info.regularizer = 'C^2 .* (t-S)^2';

    %======================================================================
    % Main loop (two d-by-n^2 GEMMs per iteration, no CPU transfers)
    %======================================================================
    for it = 1:maxIter
        C_old = C;
        S_old = S;

        % ---- 1) C update: MM upper bound + exact weighted projection ----
        XC = Xg * C;                              % d-by-n
        G  = 2 * (Xg.' * (XC - Xg));              % gradient of ||X-XC||^2
        % C^2.*(t-S)^2 is purely quadratic in C: there is no linear term.
        aC = rho + alpha * (t - S_old).^2;
        vC = (2*rho*C_old - G) ./ (2*aC);
        C  = weighted_simplex_projection(vC, aC, diagIdx);

        % ---- 2) S update: exact constrained block minimization
        %         (Gauss-Seidel, uses the new C; exact zero-weight-safe QP) ----
        % C^2.*(t-S)^2 contributes the linear term -2*alpha*t*C^2.*S.
        aS = alpha * C.^2;
        qS = P - 2*alpha*t*(C.^2);
        S  = quadratic_simplex_projection(aS, qS, diagIdx);

        % Relative iterate changes (two gathers per iteration)
        resC = sqrt(sum((C - C_old).^2, 'all')) / max(1, sqrt(sum(C.^2, 'all')));
        resS = sqrt(sum((S - S_old).^2, 'all')) / max(1, sqrt(sum(S.^2, 'all')));
        resC_g = gather(resC);  resS_g = gather(resS);
        info.resC(it) = resC_g;
        info.resS(it) = resS_g;

        % Objective value, recomputed with the current (C,S)
        if mod(it, objEvery) == 0 || it == 1
            info.obj(it) = gather(bse_objective(Xg, P, C, S, alpha, t));
        end

        if verbose && (mod(it, 100) == 0 || it == 1)
            fprintf('BSE-MM | iter %4d | obj %.6e | resC %.3e | resS %.3e\n', ...
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
        info.obj(maxIter) = gather(bse_objective(Xg, P, C, S, alpha, t));
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
% Weighted column simplex projection with double-precision internal work.
% The input/output precision is preserved. In particular, when the outer
% iteration is single precision, all quotients and cumulative sums in this
% numerically sensitive subproblem are evaluated in double precision.
    outProto = V(1);
    V = double(V);
    A = double(A);
    [n,~] = size(V);
    U = A.*V;
    U(diagIdx) = -inf;
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
    Z = cast(Z, 'like', outProto);
end

function Z = quadratic_simplex_projection(A, Q, diagIdx)
% Zero-weight-safe column simplex QP with double-precision internal work.
% No artificial floor is imposed on A: exact zeros keep their special KKT
% treatment, while arbitrarily small positive weights remain positive.
% This avoids the large single-precision roundoff caused by 1./A and
% -Q./(2*A) when A is close to zero. The output returns to the input type.
    outProto = A(1);
    A = double(A);
    Q = double(Q);
    [n,~] = size(A);
    pos = A > 0;
    pos(diagIdx) = false;
    offdiag = true(size(A), 'like', pos);
    offdiag(diagIdx) = false;
    zeroA = (~pos) & offdiag;

    % Candidate 1: all zero-weight entries are inactive. Project using only
    % strictly positive quadratic weights.
    V = zeros(size(A), 'like', A);
    invA = inf(size(A), 'like', A);
    V(pos) = -Q(pos) ./ (2*A(pos));
    invA(pos) = 1 ./ A(pos);
    U = A .* V;
    U(~pos) = -inf;
    [Us, ord] = sort(U, 1, 'descend');
    colOff = cast((0:n-1)*n, 'like', ord);
    lin = ord + colOff;
    Vs = V(lin);
    Gs = invA(lin);
    csV = cumsum(Vs, 1);
    csG = cumsum(Gs, 1);
    tauM = (csV - 1) ./ csG;
    valid = Us > tauM;
    valid(1,:) = true;
    mc = sum(valid, 1);
    tau = tauM(mc + (0:n-1)*n);
    Zp = max(0, V - tau .* invA);
    Zp(~pos) = 0;

    % Candidate 2: some zero-weight entries with the smallest linear
    % coefficient are active. The KKT multiplier then equals that minimum.
    Qzero = Q;
    Qzero(~zeroA) = inf;
    qmin = min(Qzero, [], 1);
    den = 2*A;
    den(~pos) = 1;
    Zc = max(0, (qmin - Q) ./ den);
    Zc(~pos) = 0;
    sumZc = sum(Zc, 1);
    ties = zeroA & (Q == qmin);
    nties = sum(ties, 1);
    residual = max(0, 1 - sumZc) ./ max(1, nties);
    Zc = Zc + residual .* ties;

    % If the positive-weight entries already sum to at least one at qmin,
    % Candidate 1 is optimal; otherwise Candidate 2 supplies the remainder.
    useC = any(zeroA, 1) & (sumZc < 1);
    Z = Zp;
    Z(:, useC) = Zc(:, useC);
    Z(diagIdx) = 0;
    Z = cast(Z, 'like', outProto);
end

function val = bse_objective(X, P, C, S, alpha, t)
% Objective with double-precision products/accumulations for stable logging.
% X*C-X remains in the outer precision; no objective term is altered.
    rec = X * C - X;
    val = sum(double(rec(:)).^2);
    val = val + sum(double(P(:)) .* double(S(:)));
    reg = sum(double(C(:)).^2 .* (double(t) - double(S(:))).^2);
    val = val + double(alpha) * reg;
end

function v = get_opt(opts, name, defaultValue)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = defaultValue;
    end
end