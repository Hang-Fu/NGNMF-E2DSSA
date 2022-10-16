function [oa, aa, K, ua]=confusion(true_label,estim_label)
%
% function confusion(true_label,estim_label)
%
% This function compute the confusion matrix and extract the OA, AA
% and the Kappa coefficient.
%
% INPUT
% easy! just read

l=length(true_label);  %类别标签总数
nb_c=max(true_label);  %总类数

%混淆矩阵
confu=zeros(nb_c,nb_c);
for i=1:l
  confu(estim_label(i),true_label(i))= confu(estim_label(i),true_label(i))+1;
end

%计算相应精度
oa=trace(confu)/sum(confu(:)); %overall accuracy %trace函数求矩阵的迹：矩阵的迹就是矩阵的主对角线上所有元素之和。
ua=diag(confu)./sum(confu,1)'; %class accuracy，对角线中心正确分类除以每列类别数
ua(isnan(ua))=0;               %isnan判断查询数组元素是否包含NaN值
number=size(ua,1);
aa=sum(ua)/number;             %average accuracy

Po=oa;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);  %列分别求和，行分别求和，对应类总数相乘，除以N^2，N=单元格的总数量
K=(Po-Pe)/(1-Pe);                                %kappa coefficient

%http://kappa.chez-alice.fr/kappa_intro.htm