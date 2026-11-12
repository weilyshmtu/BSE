	function [Z, S, E, time_used] = idr_gpu(X, k, gamma, lambda, opts)
	% GPU code optimized for RTX 4090D (Single Precision FP32)
	% WARNING: Using single precision may cause the convergence threshold (stopC) 
	% to plateau around 1e-3 ~ 1e-4. Adjust 'tol' in opts if full convergence is needed.
	tic
	% 1. Transfer data to GPU and force single type
	X_gpu = gpuArray(single(X));
	XtX = gpuArray(single(opts.XtX));
	I = gpuArray.eye(size(X, 2), 'single');
	% Optimization: I + ete is actually a matrix with 2 on the diagonal and 1 elsewhere.
	I_plus_ete = I + single(1); 
	% Initialize variables
	n = opts.n;
	d = opts.d;
	maxIter = opts.max_iter;
	% Note: tol = 1e-5 might be too strict for FP32. Consider relaxing to 1e-4 if it doesn't converge.
	tol = 1e-4; 
	rho = 1.1;
	max_mu = 800;
	mu = single(1e-6);
	Z = gpuArray.ones(n, n, 'single');
	S = gpuArray.zeros(n, n, 'single');
	E = gpuArray.zeros(d, n, 'single');
	C = gpuArray.ones(n, n, 'single');
	D = gpuArray.zeros(n, n, 'single');
	Y1 = gpuArray.zeros(d, n, 'single');
	Y2 = gpuArray.zeros(n, n, 'single');
	Y3 = gpuArray.zeros(1, n, 'single');
	Y4 = gpuArray.zeros(n, n, 'single');
	iter = 0;
	while iter < maxIter
	    iter = iter + 1;
	    % --- Update Z ---
	    % Optimization: Combine X_gpu'*E and X_gpu'*Y1 into a single matrix multiplication
	    temp_Y1_mE = Y1 - mu * E;
	    B_z = 2 * S + mu * XtX + X_gpu' * temp_Y1_mE;
	    A_z = 2 * I + mu * XtX;
	    Z = A_z \ B_z;
	    % --- Update S ---
	    I_C = I - C;
	    ICI_Ct = I_C * I_C';
	    A_s = 2 * (1 + mu) * I + 2 * gamma * ICI_Ct;
	    B_s = 2 * Z + mu * (C + D) - (Y2 + Y4);
	    S = (A_s \ B_s')';
	    S = max(S, 0);
	    S = 0.5 * (S + S');
	    % --- Update C ---
	    StS = S' * S;
	    A_c = 2 * gamma * StS + mu * I_plus_ete;
	    sumY3 = sum(Y3, 1); 
	    B_c = 2 * gamma * StS + mu * S + Y2 - sumY3;
	    B_c = B_c + mu;     
	    C = A_c \ B_c;  
	    % --- Update D ---
	    A_d = S + Y4 / mu;
	    t_diag = diag(A_d);    
	    eta = 2 * (k - sum(t_diag)) / n;
	    d_vec = max(0, t_diag + eta / 2); 
	    D = A_d + diag(d_vec - t_diag);
	    % --- Update E (Vectorized L21) ---
	    temp_E = X_gpu - X_gpu * Z + Y1 / mu;
	    norms_E = sqrt(sum(temp_E.^2, 1));
	    % Optimization: Use eps('single') equivalent (1e-7) to prevent division by zero in FP32
	    scales_E = max(0, 1 - (lambda / mu) ./ (norms_E + 1e-7));
	    E = temp_E .* scales_E;
	    % --- Convergence Check ---
	    leq1 = X_gpu - X_gpu * Z - E;
	    leq2 = S - C;
	    leq3 = sum(C, 1) - 1; 
	    leq4 = S - D;
	    stopC = max([max(max(abs(leq1))), max(max(abs(leq2))), ...
	                 max(max(abs(leq3))), max(max(abs(leq4)))]);
	    if mod(iter, 50) == 0 || stopC < tol
	        fprintf('iter %d, mu=%2.1e, stopALM=%2.3e\n', iter, mu, stopC);
	    end
	    if stopC < tol 
	        break;
	    end
	    Y1 = Y1 + mu * leq1;
	    Y2 = Y2 + mu * leq2;
	    Y3 = Y3 + mu * leq3; 
	    Y4 = Y4 + mu * leq4;
	    mu = min(max_mu, mu * rho);
	end
	Z = gather(Z);
	S = gather(S);
	E = gather(E);
	time_used = toc;
	end