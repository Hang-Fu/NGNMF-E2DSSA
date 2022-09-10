%==========================================================================
% G. Sun, H. Fu, et al, "A Novel Band Selection and Spatial Noise Reduction
% Method for Hyperspectral Image Classification"

% NGNMF-E2DSSA on Indian Pines dataset
%==========================================================================

close all;clear all;clc;
addpath(genpath('.\Dataset'));
addpath(genpath('.\libsvm-3.18'));
addpath(genpath('.\functions'));

%% data
load('indian_pines_gt'); img_gt=indian_pines_gt;
load('Indian_pines_corrected');img=indian_pines_corrected;
[Nx,Ny,bands]=size(img);

%% To eliminate the zero value
img2=reshape(img,Nx*Ny,bands);
img2 = img2(:);index=find(img2==0);img2(index)=0.001;
img_nor=reshape(img2,Nx*Ny,bands);
img=reshape(img_nor,Nx,Ny,bands);

tic;
%% NGNMF for band selection
%Grouping of Neighboring Bands
img2=reshape(img,Nx*Ny,bands);
img2=img2';img2 = img2(:);img2 = mapminmax(img2',0,1);  % normalization
img2(find(img2)==0)=0.001;
img_nor = reshape(img2,bands,Nx*Ny);
S = get_D(img_nor');                                    %compute similarity matrix
Sel_band = 35;                                          %selected band number
K = Sel_band + 1;
bandSubspace = subspacePart(S,bands,K);

%Band Ranking With the Normalization MF
for i=1:bands
    img3(:,:,i)=mapminmax(img(:,:,i),0,1);
end
img_m=reshape(img3,Nx*Ny,bands);
[order_band] = NMFW_RANK(img_m);
res_band_id=zeros(1,Sel_band);
for i=1:Sel_band
    [~,ind]=ismember(bandSubspace(i):bandSubspace(i+1),order_band);
    res_band_id(i)=find(ind==min(ind))+bandSubspace(i)-1;
end
img_BS=img(:,:,res_band_id);

%% E2DSSA for Spatial FE
% Spatial Binary Mask Generation
u=3; w=2*u+1; w2=w*w;                         %search window
L=25;                                         %number of embedded pixels
for i=1:bands
    img_padding(:,:,i) = padarray(img(:,:,i),[u,u],'symmetric','both');
end
moderesultAN=zeros(Nx*w,Ny*w);
for i=1:Nx
    for j=1:Ny
        i1=i+u;j1=j+u;
        testcube=img_padding(i1-u:i1+u,j1-u:j1+u,:);
        m=reshape(testcube,[w2,bands]);    
        center=m((w2+1)/2,:);NED=zeros(1,w2);
        for ii=1:w2
            NED(:,ii)=sqrt(sum(power((m(ii,:)/norm(m(ii,:))-center/norm(center)),2)));%NED
        end
        [~,ind]=sort(NED);
        index=ind(1:L);

        Mask=zeros(w2,1);
        Mask(index)=1;
        Mask=reshape(Mask,w,w);
        moderesultAN(w*(i-1)+1:w*i,w*(j-1)+1:w*j)=Mask;
    end
end

% Band-Based Spatial Processing
img_SA2DSSA=zeros(Nx,Ny,Sel_band);
img_BS_padding=img_padding(:,:,res_band_id);
for k=1:Sel_band
    X2D=zeros(L,Nx*Ny);
    ID=zeros(L,Nx*Ny);
    %Adaptive embedding
    n=0;
    for i=1:Nx
        for j=1:Ny
            n=n+1;
            i1=i+u;j1=j+u;
            test=img_BS_padding(i1-u:i1+u,j1-u:j1+u,k);
            col=moderesultAN(w*(i-1)+1:w*i,w*(j-1)+1:w*j).*test;
            col=col(:);
            sel_col=col(col~=0);
            index_col=find(col~=0);
            X2D(:,n)=sel_col;
            ID(:,n)=index_col;
        end
    end
    %Singular Value Decomposition and grouping
    S=X2D*X2D';   
    [U,autoval]=eigs(S,1);   
    V=(X2D')*U;
    rca=U*V';
    %Reprojection
    New_pad_img=zeros(Nx+w-1,Ny+w-1);
    repeat=zeros(Nx+w-1,Ny+w-1);
    kk=0;
    for i=1:Nx
        for j=1:Ny
            kk=kk+1;
            rec_col=zeros(w2,1);
            rec_col(ID(:,kk))=rca(:,kk);
            
            i1=i+u;j1=j+u;
            New_pad_img(i1-u:i1+u,j1-u:j1+u)=New_pad_img(i1-u:i1+u,j1-u:j1+u)+reshape(rec_col,w,w);
            repeat(i1-u:i1+u,j1-u:j1+u)=repeat(i1-u:i1+u,j1-u:j1+u)+moderesultAN(w*(i-1)+1:w*i,w*(j-1)+1:w*j);
        end
    end
    New_pad_img=New_pad_img./repeat;
    img_SA2DSSA(:,:,k)=New_pad_img(u+1:Nx+u,u+1:Ny+u);
end
toc;

%% training-test samples
Labels=img_gt(:);    
Vectors=reshape(img_SA2DSSA,Nx*Ny,Sel_band);  
class_num=max(max(img_gt))-min(min(img_gt));
trainVectors=[];trainLabels=[];train_index=[];
testVectors=[];testLabels=[];test_index=[];
rng('default');
Samp_pro=0.05;                                                         %proportion of training samples
for k=1:1:class_num
    index=find(Labels==k);                  
    perclass_num=length(index);           
    Vectors_perclass=Vectors(index,:);    
    c=randperm(perclass_num);                                      
    select_train=Vectors_perclass(c(1:ceil(perclass_num*Samp_pro)),:);    %select training samples
    train_index_k=index(c(1:ceil(perclass_num*Samp_pro)));
    train_index=[train_index;train_index_k];
    select_test=Vectors_perclass(c(ceil(perclass_num*Samp_pro)+1:perclass_num),:); %select test samples
    test_index_k=index(c(ceil(perclass_num*Samp_pro)+1:perclass_num));
    test_index=[test_index;test_index_k];
    trainVectors=[trainVectors;select_train];                    
    trainLabels=[trainLabels;repmat(k,ceil(perclass_num*Samp_pro),1)];
    testVectors=[testVectors;select_test];                      
    testLabels=[testLabels;repmat(k,perclass_num-ceil(perclass_num*Samp_pro),1)];
end
[trainVectors,M,m] = scale_func(trainVectors);
[testVectors ] = scale_func(testVectors,M,m);   

%% SVM-based classification
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
models=svmtrain(trainLabels,trainVectors,cmd);
testLabel_est= svmpredict(testLabels,testVectors, models);

%classification map
result_gt= Labels;       
for i = 1:1:length(testLabel_est)        
   result_gt(test_index(i)) = testLabel_est(i);  
end
result_map_l = reshape(result_gt,Nx,Ny);result_map=label2color(result_map_l,'india');figure,imshow(result_map);

%classification results
[OA,AA,kappa,CA]=confusion(testLabels,testLabel_est);
result=[CA*100;OA*100;AA*100;kappa*100]


