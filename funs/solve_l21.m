function [E] = solve_l21(W, lambda)
% SOLVE_L21 求解 L2,1 范数最小化问题
% min_E lambda * ||E||_{2,1} + 0.5 * ||E - W||_F^2
% 按列处理，||E||_{2,1} 表示各列 L2 范数之和

    n = size(W, 2);
    E = zeros(size(W));
    
    for i = 1:n
        col_norm = norm(W(:, i), 2);
        if col_norm > lambda
            % Shrinkage 操作
            E(:, i) = (col_norm - lambda) / col_norm * W(:, i);
        else
            E(:, i) = 0; % 设为全 0 列 (稀疏诱导)
        end
    end
end