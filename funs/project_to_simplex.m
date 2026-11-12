function v_proj = project_to_simplex(v)
% PROJECT_TO_SIMPLEX Projects a vector onto the probability simplex
%
% Inputs:
%   v -- n×1 column vector, the vector to be projected
%
% Outputs:
%   v_proj -- n×1 column vector, the projected vector satisfying sum(v_proj) = 1 and v_proj >= 0
%
% Algorithm reference:
%   Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). 
%   Efficient projections onto the l1-ball for learning in high dimensions.
%   ICML 2008.

% Ensure input is a column vector
v = v(:);
n = length(v);

% Step 1: Sort the vector in descending order
u = sort(v, 'descend');

% Step 2: Compute prefix sums and find the optimal threshold
% Find the largest k such that u(k) - (sv(k) - 1)/k > 0
% Use vectorized operation to avoid loops
sv = cumsum(u);
rho = find(u - (sv - 1)./(1:n)' > 0, 1, 'last');

% Step 3: Compute the threshold theta
theta = (sv(rho) - 1) / rho;

% Step 4: Truncate and output
v_proj = max(v - theta, 0);

% Numerical stability check: Ensure sum is 1 (avoid floating-point errors)
v_proj = v_proj / sum(v_proj);
end