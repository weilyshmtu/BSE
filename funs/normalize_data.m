function nX = normalize_data(X, ntype)
nX = zeros(size(X));
    switch ntype
        case 1
            nX = X ./ (sqrt(sum(X.^2, 1)) + 1e-12);
        case 2
            [~, n] = size(X);
            for j = 1:n
                normItem = std(X(:,j));
                if (0 == normItem)
                    normItem = eps;
                end
                nX(:,j) = (X(:,j) - mean(X(:,j)))/normItem;
            end
        case 3
            X = X - min(X(:));
            nX = X./max(max(X(:))+eps);
            % nX = X./repmat((255)*ones(1,size(X,2)),size(X,1),1);
    end
end