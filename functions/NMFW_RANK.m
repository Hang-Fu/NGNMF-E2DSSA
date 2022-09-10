function [order_band] = NMFW_RANK(pre)
%pre 是N*B大小的数据，N是像素数，B是波段数；order_band是排序结果
% informaation content
m=mean(pre);
K=cov(pre);K_1=inv(K);
k=1./sum((((pre-m)*K_1).*(pre-m)),2); %Eq7
w=k'.*(K_1*(pre-m)');%Eq1

w=abs(w);
rho=mean(w,2);       %Eq10
[~,order_band] = sort(rho,'descend');
% [~,order_band] = sort(rho);
end