function [C, W, labels, info] = lsr_gpu(X, delta, opts)
%==========================================================================
% LSR baseline for ablation experiments (GPU MATLAB).
%
% Input X is d-by-n: each COLUMN is one sample.
%
% Model:
%   min_C ||X-X*C||_F^2 + delta*||C||_F^2.
% Closed form:
%   C = (X'*X + delta*I_n) \ (X'*X)
%     = X' * ((X*X' + delta*I_d) \ X).
% The smaller of the d-by-d and n-by-n linear systems is solved.
%
% Standalone LSR postprocessing (not BSE initialization):
%   C_ii <- 0, A = abs(C), W = (A+A')/2,
%   normalized spectral clustering on W.
%
% Usage:
%   [C,W,labels,info] = LSR_gpu(X,0.001,c,struct());
%
% Optional opts:
%   precision : 'double' (default) or 'single'
%   seed      : RNG seed for k-means; default 1
%   replicates: k-means replicates; default 10
%   verbose   : default true
%
% Outputs C and W are n-by-n CPU matrices. labels is n-by-1.
% Requires spectral_clustering_gpu.m on the MATLAB path.
%==========================================================================
    tic;
    if nargin < 3 || isempty(opts), opts = struct(); end
    if ndims(X) ~= 2 || ~isnumeric(X)
        error('X must be a numeric d-by-n matrix.');
    end
    if ~isscalar(delta) || ~isfinite(delta) || delta <= 0
        error('delta must be a finite positive scalar.');
    end
    d = opts.d;
    n = opts.n;


    prec = get_opt(opts,'precision','double');
    verbose = get_opt(opts,'verbose',true);
    if ~ismember(prec,{'single','double'})
        error('opts.precision must be ''single'' or ''double''.');
    end

    Xg = full(gpuArray(cast(X,prec)));
    deltaG = cast(delta,'like',Xg);

    % Solve the smaller equivalent linear system; do not form inv(.).
    if d <= n
        M = Xg*Xg.' + deltaG*eye(d,'like',Xg);
        Cg = Xg.' * (M\Xg);
        info.solve_size = d;
    else
        K = Xg.'*Xg;
        Cg = (K + deltaG*eye(n,'like',Xg))\K;
        info.solve_size = n;
        clear K;
    end

   
    C = gather(Cg);
    info.delta = delta;
    info.time = toc;
    if verbose
        fprintf('LSR-GPU | delta %.4g | solve %d-by-%d | time %.3f s\n', ...
                delta,info.solve_size,info.solve_size,info.time);
    end
end

function v = get_opt(opts,name,defaultValue)
    if isfield(opts,name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = defaultValue;
    end
end
