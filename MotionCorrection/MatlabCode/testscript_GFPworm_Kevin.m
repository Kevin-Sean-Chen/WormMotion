%testscript_GFPworm_Kevin
%% load file
%%%moving GFP
file = '/tigress/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_155051/';%BrainScanner20160506_160928/';%';%';  %%%GFP worms

%%%moving GCaMP
% file = '/tigress/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620/';
% array(['BrainScanner20170424_105620', 'BrainScanner20170610_105634',
%        'BrainScanner20170613_134800', 'BrainScanner20180709_100433',
%        'BrainScanner20181129_120339'], dtype='<U27')

%imobolized GFP
% file = '/tigress/LEIFER/PanNeuronal/2018/20180518/BrainScanner20180518_093125/'; %BrainScanner20180518_091402 %BrainScanner20180518_094052

%% imaging data
load([file,'heatData.mat']);
remove = 200;
GFP = gPhotoCorr(:,remove:end);
RFP = rPhotoCorr(:,remove:end);
ng = GFP;
nr = RFP;
ng(isnan(GFP)) = 0;
nr(isnan(RFP)) = 0;

%%
subplot(131);imagesc(ng)
subplot(132);imagesc(nr)
subplot(133);imagesc(ng./nr)
%% smoothing test
%the idea is to smooth both traces before taking ratiometric
smooG = 5;
smooR = 10;
sG = zeros(size(ng));
sR = zeros(size(nr));
for nn = 1:size(ng,1)
    sG(nn,:) = smooth(ng(nn,:),smooG);
    sR(nn,:) = smooth(nr(nn,:),smooR);
end
subplot(141);imagesc(sG); subplot(142);imagesc(sR)
subplot(143);imagesc(ng./nr); subplot(144); imagesc(sG./sR)

%% load behavioral variables
load([file,'heatDataMS.mat']);
PC1 = behavior.pc1_2(:,1);
PC2 = behavior.pc1_2(:,2);

%% regress out correlated signal first
%...can include map correction and smoothing...
Ng = zeros(size(ng));
Nr = zeros(size(nr));
for ii = 1:size(Ng,1)
    S = ng(ii,:)';
    X = [ones(length(S),1)];% mean(ng,1)' PC1(remove:end)];%
    [b,bint,r_,rint,stats] = regress(S,X(:,:));
    Ng(ii,:) = r_;%S - X*b; %residuals of the predictors
    S = nr(ii,:)';
    X = [ones(length(S),1)];% mean(nr,1)' PC1(remove:end)];%
    [b,bint,r_,rint,stats] = regress(S,X(:,:));
    Nr(ii,:) = r_;
end

%% select single neuron from data
nrn  = 10; % just pick an index
T = 300; % set shorter time series
k = 300;
rr = Ng(nrn,1+k:T+k)';
gg = Nr(nrn,1+k:T+k)';
rr(isnan(rr)) = nanmean(rr);  %%%removing Nan effects
gg(isnan(gg)) = nanmean(gg);  %%%removing Nan effects

% standardize rr and gg
rr = rr./nanstd(rr);  %
gg = gg./nanstd(gg);  %

figure;
subplot(211); plot(rr); title('measured motion artifact and rfp')
subplot(212); plot(gg); title('measured gcamp')

alpha_g = 0.7; % single time-bin decay of gcamp fluorescence signal
tau_g = -1/log(alpha_g);  % time constant of gcamp decay (in time bins) 
D = spdiags(ones(T,1)*[-alpha_g 1],-1:0,T,T); % better implementation!
%% Hand-tune the GP params (bye eye)
% Compute autocovariance 
nlags = 500;
xx = -nlags:nlags;

xcsamp = xcov(rr-nanmean(rr),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');

% Define true GP covariance function
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
      
rho_m = 0.1*10+0.1;  % prior variance of m
lm = 8*1; % length scale
pm = 1.5*1; % power  (note: only valid covariance function for p<=2)
sigr = 1;

kfunplot = kfun(xx,rho_m,lm,pm,sigr);
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');


%% 2. find ML estimates of the GP prior over motion, variances for r and g
prs0 = [rho,lm,pm,sigr]'; % "true" params
LB = [.01,.5, .1, .01]'; % lower bound
UB = [2*var(rr),2*length(rr),2,std(rr)]'; % upper bound

% Center the data so we don't need to fit the mean
muEst_m = mean(rr);
rr_ctr = rr-muEst_m;
% anonymous function for estimating kernel covariance
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
% make anonymous function (function pointer) for neg log-likelihood function.
lfun = @(prs)neglogli_GP(prs,kfun,rr_ctr);

% Set initial params
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB


% set optimization options
opts = optimset('display','iter');

% optimize
prsML = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);
sigEst_r = prsML(4);
% find mu_y estimate
muEst_a = mean(gg)/muEst_m;
muEst_y = D*(muEst_a.*ones(T,1)); % can the first few elements ever be estimated correctly? does aa = D\yy always result in non-stationarity? 
rho_a = 1.1;%;var(gg);  % variance of a(t) 
% set prior covariance for neural activity by hand 


% compute estimates of full covariances etc needed later in motion
% correction


% use estimated parameter values
KmEst = toeplitz(prsML(1)*exp(-abs((1:T)/prsML(2)).^prsML(3)));
[UmEst,SmEst] = svd(KmEst);
thresh = 1e12;  % threshold on condition number
smdiagEst = diag(SmEst); 
ii = max(smdiagEst)./smdiagEst < thresh;  % vector of indices to keep.
krankEst = sum(ii); % rank
UbasisEst = UmEst(:,ii);  % basis for Km
SminvEst = spdiags(1./smdiagEst(ii),0,krankEst,krankEst); 

% find estimate of sig_g from data
prs0 = [1 1 2 std(gg)]';
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB

lfuncG = @(prs)neglogli_GP(prs,kfun,gg - mean(gg));
prsMLG = fmincon(lfuncG,prs0,[],[],[],[],LB,UB,[],opts);
sigEst_g = prsMLG(4);

%% 3. Do motion correction
% aa0 = gg./rr;
% aa0(isnan(aa0)) = nanmean(aa0);  %%remove 0/0 effects due to original Nan values
% yy0 = D*aa0;
% sigEst_y = 1;
% % sigEst_g = 10;
% % sigEst_r = 10;
% yyEst = MotionCorrection_smooth_yy(rr,gg,aa0,sigEst_r^2,sigEst_g^2,muEst_y,sigEst_y,muEst_m,UbasisEst,SminvEst,D);
% aaEst = D\yyEst;  % corresponding estimate of y(t)

%% %%% faster methods
%% Estimate a(t) with iid priors on m(t) and a(t) (JP code)
aa0 = gg./rr;
% aa0(isnan(aa0)) = 0;
aa1 = MotionCorrection_iid(rr,gg,sigEst_r^2,sigEst_g^2,muEst_m,rho_m,muEst_a,rho_a);
yy1 = D*aa1; % corresponding estimate of y(t)

%% 4. find fluorescence a(t), smooth prior (JP code).

Sminv = spdiags(1./smdiagEst(ii),0,krankEst,krankEst); % diagonal matrix sqrt of eigenvalues
aa3 = MotionCorrection_smooth_aa(rr,gg,aa0,sigEst_r^2,sigEst_g^2,muEst_a,rho_a,muEst_m,UbasisEst,Sminv,D);
yy3 = D*aa3;  % corresponding estimate of y(t)

%% evaluate estimates vs g/r

subplot(211);  % a(t) plotsfi
plot(1:T,gg);
hold on
plot(1:T,rr)
subplot(212); 
plot(1:T, [aa0 aa1,aa3]);%[aa0 aaEst]); 
title('estimated a(t)');
legend('g/r', ' motion corrected (iid)','smooth a');
  