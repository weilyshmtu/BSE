function [Z,S,E,time_used] = idr_cpu(X,k,gamma,lambda, opts)

% min ||Z - S||_F^2 + gamma||S||_id + lambda||E||_2,1
% s.t. X = XZ + E, e'S = e, S= S', S>=0, tr(S) = k
% 
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors, k is the number of subspaces

tic
tol = 1e-5;
maxIter = opts.max_iter; 
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

d = opts.d; 
n = opts.n;
e = opts.e';
I = eye(n);
XtX = opts.XtX;
ete = opts.eet';

Z = ones(n,n);
S = zeros(n,n);
E = zeros(d,n);
C = ones(n,n);
D = zeros(n,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = zeros(1,n);
Y4 = zeros(n,n);

iter = 0;
while iter < maxIter
    iter = iter + 1;
    % preZ = Z; preS = S; preE = E;
    
    % update Z
    A_z = 2*I + mu*XtX;
    B_z = 2*S + mu*(XtX - X'*E) + X'*Y1;
    Z = A_z \ B_z;
    
    % update S
    I_C = I - C;
    A_s = 2*(1+mu)*I + 2*gamma*(I_C * I_C');
    B_s = 2*Z + mu*C - Y2 + mu*D - Y4;
    S = B_s / A_s;
    S = max(S, 0);
    S = 0.5*(S + S');
    
    % update C
    A_c = 2*gamma*(S'*S) + mu*(I + ete);
    B_c = 2*gamma*(S'*S) + mu*S + Y2 + mu*ete - e'*Y3;
    C = A_c \ B_c;  
    
    % update D
    A_d = S + Y4/mu;
    t_diag = diag(A_d);    
    D_mat = A_d - diag(t_diag);
    eta = 2*(k - sum(t_diag))/n;
    d_vec = max(0, t_diag + eta/2); 
    D = D_mat + diag(d_vec);
    
    % update E (L21 Norm)
    temp_E = (X - X*Z) + Y1/mu;
    E = solve_l1l2(temp_E, lambda/mu);
    
    % Convergence Check
    leq1 = X - X*Z - E;
    leq2 = S - C;
    leq3 = e*C - e;
    leq4 = S - D;
    stopC = max([max(abs(leq1(:))), max(abs(leq2(:))), max(abs(leq3(:))), max(abs(leq4(:)))]);
    
    if mod(iter, 50) == 0 || stopC < tol
        fprintf('iter %d, mu=%2.1e, stopALM=%2.3e\n', iter, mu, stopC);
    end
    
    if stopC < tol 
        time_used = toc;
        break;
    end
    
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;
    Y4 = Y4 + mu*leq4;
    mu = min(max_mu, mu*rho);
end
if iter >= maxIter
    time_used = toc;
end

end



function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end