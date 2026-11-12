function [C, F, time_used] = sdsgc_cpu(X, k, opts)
%% min_{W>=0,W1=1,W=W'}sum_{i,j=1}^n||x_i-x_j||_2^2w_{ij}+gamma||W||_F^2+2*lambda*trace(F'*L(W)*F)
% X: dim*num data matrix, each column is a data point
% k: number of neighbors to determine the initial graph
% W: num*num the learned doubly stochastic  matrix
% local:capture localized data relationships by enforcing the sparsity of the graph
% local= 2;
tic
maxIter = 300;
n = opts.n;
c = opts.n_clusters;
Dx = opts.Dx;
[~, idx] = sort(Dx, 2);

C = opts.S_init;
D = diag(sum(C));
L = D - C;
[F, ~, evs]=eig1(L, c, 0,0);
tol = 1e-10;
% if sum(evs(1:c+1)) < tol
%     error('The original graph has more than %d connected component', c);
% end

rho = 1.3;
mu = 1;
eta = 10;
local = 1;
% [Para,idx] = symNeighbors(distX,k);
LAMBDA = zeros(n);

gamma = 0;
for i = 1:n
    gamma = gamma + (k/2*Dx(i,k+2) - 1/2*sum(Dx(i,idx(i,2:k+1))));
end
gamma = gamma/n;

for iter = 1:maxIter

    % update W
    distF = L2_distance_1(F',F');
    Z = mu*C'-1/2*LAMBDA+1/2*LAMBDA'-1/2*Dx-1/2*eta*distF;
    C = zeros(n);
    for i=1:n
        if local == 1
          % idxa0 = Para.kn{i};
          idxa0 = idx(i,2:k);
        else
          idxa0 = idx(i,2:n);
        end
        ad = Z(i,idxa0)/(gamma+mu);
        C(i,idxa0) = EProjSimplex_new(ad);    
    end

    % Update LAMBDA
    h = C-C';
    LAMBDA = LAMBDA+mu*h;

    % Update mu
    mu = rho*mu;

    % rank constraint
    C = (C+C')/2;
    D = diag(sum((C+C')/2));
    F_old = F;
    [F, ~, ev]=eig1(D-C, c, 0);
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1)); 

    flag = sum(logical(diag(D-eye(n))<0.01)); % check  whether the graph is doubly stochastic
    if fn1 > tol 
        eta = 2*eta;
    elseif fn2 < tol
        eta = eta/2;  
        F = F_old;
    elseif fn1 < tol && fn2 > tol && flag == n  % output graph in advance [when the graph is structured and doubly stochastic] 
        time_used = toc;
        break;
    end

end

if iter >= maxIter
    time_used = toc;
end