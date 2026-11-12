function sig2 = spectral_rho(X)
% Estimate sigma_1(X)^2 = lambda_max(X'X) by power iteration.
% X is d-by-n; the n-by-n Gram matrix is NEVER formed.
    [~,n] = size(X);
    v = randn(n,1,'like',X);  v = v/norm(v);
    sig2 = cast(0,'like',X);
    for it = 1:40
        u  = X*v;            % d×1
        w  = X.'*u;          % n×1 = (X'X) v
        v  = w/norm(w);
        s2 = sum(u.^2);      % Rayleigh 商 v'(X'X)v = ||Xv||^2
        if abs(s2-sig2)/max(s2,eps) < 1e-6, break; end
        sig2 = s2;
    end
end
