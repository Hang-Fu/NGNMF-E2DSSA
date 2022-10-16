function [oa, aa, K, ua]=confusion(true_label,estim_label)
%
% function confusion(true_label,estim_label)
%
% This function compute the confusion matrix and extract the OA, AA
% and the Kappa coefficient.
%
% INPUT
% easy! just read

l=length(true_label);  %����ǩ����
nb_c=max(true_label);  %������

%��������
confu=zeros(nb_c,nb_c);
for i=1:l
  confu(estim_label(i),true_label(i))= confu(estim_label(i),true_label(i))+1;
end

%������Ӧ����
oa=trace(confu)/sum(confu(:)); %overall accuracy %trace���������ļ�������ļ����Ǿ�������Խ���������Ԫ��֮�͡�
ua=diag(confu)./sum(confu,1)'; %class accuracy���Խ���������ȷ�������ÿ�������
ua(isnan(ua))=0;               %isnan�жϲ�ѯ����Ԫ���Ƿ����NaNֵ
number=size(ua,1);
aa=sum(ua)/number;             %average accuracy

Po=oa;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);  %�зֱ���ͣ��зֱ���ͣ���Ӧ��������ˣ�����N^2��N=��Ԫ���������
K=(Po-Pe)/(1-Pe);                                %kappa coefficient

%http://kappa.chez-alice.fr/kappa_intro.htm