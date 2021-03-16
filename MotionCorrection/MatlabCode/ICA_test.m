%%%ICA_test
%from fast-ICA package: https://github.com/davidkun/FastICA
clear
clc
%%
nTRs=1000; 
%number of timepoints in the fMRI timecourse
nComps=3;  
%number of groundtruth components
%put together synthetic data (2D x time)
t=1:nTRs;  
pS=5;     
%patch size
iS=40;    
%image size
I{1}=zeros(iS,iS); I{1}(1:1+pS,1:1+pS)=1;     
T{1}=gamrnd(10,5,1,1,nTRs);
I{2}=zeros(iS,iS); I{2}(11:11+pS,11:11+pS)=1; 
T{2}=gamrnd(5,3,1,1,nTRs);
I{3}=zeros(iS,iS); I{3}(21:21+pS,21:21+pS)=1; 
T{3}=gamrnd(3,2,1,1,nTRs);
figure(1);
set(gcf,'Name','Ground truth components');

subplot(2,3,1); imagesc(I{1}); 
subplot(2,3,2); imagesc(I{2}); 
subplot(2,3,3); imagesc(I{3}); 
subplot(2,3,4); plot(squeeze(T{1})); 
subplot(2,3,5); plot(squeeze(T{2})); 
subplot(2,3,6); plot(squeeze(T{3})); 
data=zeros(iS,iS,nTRs);
for iter=1:nComps
    data=data + repmat(I{iter},[1 1 nTRs]).*repmat(T{iter},[iS iS 1]);
end

%% PCA/SVD decomposition of the data
[U,S,V]=svd(reshape(data,[iS*iS nTRs]),'econ');
%show recovered components
for iter=1:nComps
    C_PCA{iter}=reshape(U(:,iter),[iS iS]);
end

figure(2);
set(gcf,'Name','SVD results');
subplot(2,3,1); imagesc(C_PCA{1}); 
subplot(2,3,2); imagesc(C_PCA{2}); 
subplot(2,3,3); imagesc(C_PCA{3}); 
subplot(2,3,4); plot(squeeze(V(:,1))); 
subplot(2,3,5); plot(squeeze(V(:,2))); 
subplot(2,3,6); plot(squeeze(V(:,3))); 

%% FastICA (data according to convention observation=row)
GR_data = [ng(10,:); nr(10,:)];
nComps = 2;
[icasig, A, W] = fastica(GR_data, 'lastEig', 2,'numOfIC', nComps);
%show recovered components
for iter=1:nComps
    C{iter}=reshape(icasig(iter,:),[iS iS]);
end

figure(3);
set(gcf,'Name','ICA results');
subplot(2,3,1); imagesc(C{1}); 
subplot(2,3,2); imagesc(C{2}); 
subplot(2,3,3); imagesc(C{3}); 
subplot(2,3,4); plot(squeeze(W(1,:))); 
subplot(2,3,5); plot(squeeze(W(2,:))); 
subplot(2,3,6); plot(squeeze(W(3,:))); 

%% Worm data!!!
temp_r = nr(1:10,1:1500);
temp_g = ng(1:10,1:1500);
[icasig, A, W] = fastica([temp_g; temp_r],'numOfIC', nComps);
subplot(211)
plot(icasig')
subplot(212)
plot(temp_g'); hold on; plot(temp_r')

%% loop for all cells
ICA_g = [];
ICA_r = [];
for ii = 1:size(ng,1)
    temp_r = nr(ii,1:1000);
    temp_g = ng(ii,1:1000);
    [icasig, A, W] = fastica([temp_g; temp_r],'numOfIC', nComps,'displayMode', 'off');
    ICA_g = [ICA_g; icasig(1,:)];
    ICA_r = [ICA_r; icasig(2,:)];
end

%% siding window
win = 500;
cellID = 15;
ICA_g = [];
ICA_r = [];
for tt = 1:size(ng,2)-win
    temp_r = nr(cellID,tt:tt+win);
    temp_g = ng(cellID,tt:tt+win);
    [icasig, A, W] = fastica([temp_g; temp_r],'numOfIC', nComps,'verbose', 'off');
    ICA_g = [ICA_g; icasig(1,end)];
    ICA_r = [ICA_r; icasig(2,end)];
end

subplot(211)
plot(ICA_g); hold on; plot(ICA_r)
subplot(212)
plot(ng(cellID,:)); hold on; plot(nr(cellID,:))

