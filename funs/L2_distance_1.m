% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function D = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
D = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

D = real(D);
D = max(D,0);

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end
