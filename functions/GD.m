function [ S ] = GD(X)
    % Normalization, which was not mentioned in the original paper
    % for i = 1 : L
    %     X(:, i) = X(:, i) / norm(X(:, i));
    % end
    S = sqrt(L2_distance(X));
end

% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

function d = L2_distance(a)
    sm=ones(1,size(a,1));
    aa=sm*(a.*a);  ab=(a'*a); 
    aa = round(10000*aa)/10000;    ab = round(10000*ab)/10000;     %削弱matlab计算精度引起的误差  
    d = repmat(aa',[1 size(aa,2)]) + repmat(aa,[size(aa,2) 1]) - 2*ab;
    d = real(d);
    d = max(d,0);
end