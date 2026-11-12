function [S, info] = CAN_gpu(X, opts)
%==========================================================================
% Original adaptive-neighbor graph construction (CAN), GPU MATLAB.
% No F update, no Laplacian rank constraint, no iteration, and no internal
% clustering. The only algorithmic output is the adjacency matrix S.
%
% Input:
%   X : d-by-n data matrix; each COLUMN is one sample.
%
% opts (optional):
%   k0        : number of adaptive neighbors, default 10
%   precision : 'single' (default) or 'double'
%
% Output:
%   S    : n-by-n nonnegative adjacency matrix with zero diagonal and
%          unit COLUMN sums, matching the BSE interface.
%   info : k0, degenerateRows, time (diagnostics only).
%
% Construction:
%   P_ij = ||x_i-x_j||_2^2. For each sample i, exclude x_i itself and let
%   Omega_i contain its k0 nearest neighbors. Define d_i as the distance
%   to its (k0+1)-st nearest neighbor. The row-stochastic graph S~ is
%
%     S~_ij = (d_i-P_ij)/(k0*d_i-sum_{l in Omega_i} P_il), j in Omega_i,
%     S~_ij = 0,                                               otherwise.
%
%   Finally S = S~.', so column j contains sample j's neighbor weights.
%
% Degenerate rows with a zero denominator (caused by tied/duplicate
% distances) receive uniform weight 1/k0 on their selected k0 neighbors.
% This deterministic fallback introduces no additional parameter.
%
% Example:
%   opts = struct('k0',10,'precision','single');
%   [S,info] = CAN_gpu(X,opts);
%   labels = SpectralClustering(S+S.',c);  % external, as for BSE
%==========================================================================
    tic;
    if nargin < 2 || isempty(opts), opts = struct(); end
    if ndims(X) ~= 2 || ~isnumeric(X)
        error('X must be a numeric d-by-n matrix.');
    end
    n = size(X,2);
    k0 = get_opt(opts,'k0',10);
    prec = get_opt(opts,'precision','single');

    if ~ismember(prec,{'single','double'})
        error('opts.precision must be ''single'' or ''double''.');
    end
    if ~isscalar(k0) || ~isfinite(k0) || k0 < 1 || ...
            k0 > n-2 || k0 ~= floor(k0)
        error('opts.k0 must be an integer in [1,n-2].');
    end

    Xg = full(gpuArray(cast(X,prec)));

    % Squared Euclidean distances between all columns of X.
    x2 = sum(Xg.^2,1);
    P = x2.'+x2-2*(Xg.'*Xg);
    P = max(P,0);
    P = (P+P.')/2;

    % Exclude self-distances, then sort each row. The first k0 entries are
    % neighbors; entry k0+1 supplies the adaptive-neighbor threshold.
    diagIdx = 1:(n+1):(n*n);
    P(diagIdx) = cast(inf,'like',P);
    [Dsort,ord] = sort(P,2,'ascend');
    Dk = Dsort(:,1:k0);
    dk1 = Dsort(:,k0+1);
    neighborIdx = ord(:,1:k0);

    den = k0*dk1-sum(Dk,2);
    valid = isfinite(den) & (den > 0);
    denSafe = den;
    denSafe(~valid) = 1;  % temporary value; replaced below

    weights = (dk1-Dk)./denSafe;
    uniformWeight = cast(1/k0,'like',P);
    weights(~valid,:) = uniformWeight;

    % Scatter each row's k0 weights into its selected neighbor positions.
    Srow = zeros(n,n,'like',P);
    rowIdx = cast((1:n).','like',ord);
    linearIdx = rowIdx+(neighborIdx-1)*n;
    Srow(linearIdx) = weights;
    Srow(diagIdx) = 0;

    % Remove only floating-point row-sum error; the exact formula already
    % gives unit row sums for all nondegenerate rows.
    Srow = Srow./sum(Srow,2);
    S = gather(Srow.');
    S(diagIdx) = 0;

    info.k0 = k0;
    info.degenerateRows = gather(sum(~valid));
    info.time = toc;
end

function v = get_opt(opts,name,defaultValue)
    if isfield(opts,name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = defaultValue;
    end
end
