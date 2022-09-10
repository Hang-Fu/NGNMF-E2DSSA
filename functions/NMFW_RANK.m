function [order_band] = NMFW_RANK(pre)
%pre ��N*B��С�����ݣ�N����������B�ǲ�������order_band��������
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