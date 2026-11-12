function vec = ISC_ind2vec(ind)
maxInd = max(ind);
vec = zeros(maxInd, length(ind));
for i = 1 :length(ind)
    vec(ind(i), i) = 1;
end