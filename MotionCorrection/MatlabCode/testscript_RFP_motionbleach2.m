%testscript_RFP_motionbleach
%
%% load file
%imobolized GFP
file = '/tigress/LEIFER/PanNeuronal/2018/20180518/BrainScanner20180518_093125/'; %BrainScanner20180518_091402 %BrainScanner20180518_094052
file_imm = '/tigress/LEIFER/PanNeuronal/2017/20171128/BrainScanner20171128_154753/'; %%%very immobalized GFP worm
load([file_imm,'heatData.mat'])
FF = rRaw;

%%%moving GFP
file_mov = '/tigress/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_155051/';%BrainScanner20160506_160928/';%';%';  %%%moving GFP worms
load([file_mov,'heatData.mat'])
RR = rRaw;
GG = gRaw;

% %%%moving GCaMP
% file = '/tigress/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620/';
% load([file_mov,'heatData.mat'])
% RR = rRaw;
% GG = gRaw;

%% test code for decay form
cellID = 10;
rr = rRaw(cellID,:);

%exponential form
Fexponent = fittype('a*exp(b*x)+c','dependent',{'y'},'independent',...
{'x'},'coefficients',{'a', 'b', 'c'});
%test with doulbe exponent
% Fexponent = fittype('a*exp(b*x)+c*exp(d*x)+e','dependent',{'y'},'independent',...
% {'x'},'coefficients',{'a', 'b', 'c', 'd', 'e'});

present = (~isnan(rr)');
present = present & (rr~=0)';
xVals = (1:length(rr))';
xVals = xVals(present);
rVals = rr';
rVals = rVals(present);
rVals = rVals/max(rVals);  %normalization

%all the fitting options required
minWindow=150;
fitOptions=fitoptions(Fexponent);
fitOptions.Lower=[0,-.2,0];
fitOptions.Upper=[1000,0,10000];
fitOptions.StartPoint=[range(rVals(rVals~=0)),-.0006,min(rVals(rVals~=0))];
%%%for double-exponentials
% fitOptions.StartPoint=[range(rVals(rVals~=0)),-.0006,range(rVals(rVals~=0)),-.0006,min(rVals(rVals~=0))];
fitOptions.Weights=zeros(size(rVals));
fitOptions.Weights(minWindow:end-minWindow)=1;
        
%do exponential fitting
[f,fout]=fit(xVals,rVals,Fexponent,fitOptions);

%% test for decay stability
dir = '/tigress/LEIFER/PanNeuronal/2018/20180518/'; %BrainScanner20180518_091402 %BrainScanner20180518_094052
files = {'BrainScanner20180518_093125/', 'BrainScanner20180518_091402/', 'BrainScanner20180518_094052/'};
cc = {'r','g','b'};
minWindow=150;
for ff = 1:length(files)
    load([dir, files{ff}, 'heatData.mat'])
    rVals = nanmean(rRaw)';
    xVals = (1:length(rVals))';
    present = (~isnan(rVals));
    rVals = rVals(present);
    rVals = rVals/max(rVals);
    plot(rVals,cc{ff})
    hold on
    xVals = xVals(present);
    fitOptions.Weights=zeros(size(rVals));
    fitOptions.Weights(minWindow:end-minWindow)=1;
    [f,fout]=fit(xVals,rVals,Fexponent,fitOptions);
    plot(f.a*exp(f.b*xVals)+f.c,'--')
    %%%double-exponentials
%     plot(f.a*exp(f.b*xVals)+f.c*exp(f.d*xVals)+f.e,'k--')%[cc{ff},'--'])
    f
end
%legend('','\tau_1=2.4832e+05, \tau_2=-1.7519e+03','','-5.9737e+03, -1.4302e+03','','-5.1203e+03, -1.2412e+03')

%% expnontial decay correction with normalization 
% file_mov = '/tigress/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_155051/';%BrainScanner20160506_160928/';%';%';  %%%moving GFP worms
file = '/tigress/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620/'; %%%moving GCaMp
load([file_mov,'heatData.mat'])
RR = rPhotoCorr;%rRaw;
GG = gPhotoCorr;%gRaw;

corrected_R = zeros(size(RR));
corrected_G = zeros(size(GG));
for nn = 1:size(RR,1)
    
    xVals=(1:size(rRaw,2))';
    % only take values where bot R and G are present
    present = (~isnan(rRaw(nn,:)+gRaw(nn,:))');
    present = present & (rRaw(nn,:)~=0)' & (gRaw(nn,:) ~=0)';
    xVals = xVals(present);
    % get R and G traces
    rVals = rRaw(nn,:)';
    gVals = gRaw(nn,:)';
    gVals = gVals(present);
    denom_g = gVals(1);
    gVals = gVals/denom_g;%max(gVals);
    rVals = rVals(present);
    denom_r = rVals(1);
    rVals = rVals/denom_r;%max(rVals);

    fitOptions.Weights=zeros(size(rVals));
    fitOptions.Weights(minWindow:end-minWindow)=1;
    [f,fout]=fit(xVals,rVals,Fexponent,fitOptions);
    reconstruct_bleech_r = f.a*exp(f.b*xVals)+f.c;
    [f,fout]=fit(xVals,gVals,Fexponent,fitOptions);
    reconstruct_bleech_g = f.a*exp(f.b*xVals)+f.c;
    corrected_R(nn,1:length(rVals)) = denom_r*(rVals - reconstruct_bleech_r);
    corrected_G(nn,1:length(gVals)) = denom_g*(gVals - reconstruct_bleech_g);
    
end

%% imaging data
remove = 100;
lim = min(size(FF,2),size(RR,2));
nFF = FF(:,remove:lim);
nRR = RR(:,remove:lim);
nFF(nFF>2000) = nanmean(nanmean(nFF));
nRR(nRR>2000) = nanmean(nanmean(nRR));
nFF(isnan(nFF)) = nanmean(nanmean(nFF));  %roughly remove them for now (??)
nRR(isnan(nRR)) = nanmean(nanmean(nRR));
subplot(121); imagesc(nFF); subplot(122); imagesc(nRR);

%% load behavioral variables
load([file_mov,'heatDataMS.mat']);
PC1 = behavior.pc1_2(:,1);
PC2 = behavior.pc1_2(:,2);

%% local density
%% CMOS space
%% scanning artifact
%% global change
fluo = nanmean(nRR,1);
%% setup initial values/priors
cellID = 10;
ff = nanmean(nFF,1);  %the way to measure FF (??)
rr = smooth(nRR(cellID,:),5);
X = [ones(length(fluo),1)  PC1(remove:end) ff']';
T = 1:length(rr);

%% GP prior on parameters
%%%%%%%%%%%
%% check covariance
nlags = 300;
xx = -nlags:nlags;
xcsamp = xcov(rr-nanmean(rr),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');

% Define true GP covariance function
% kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
kfun = @(x,r,l,p,sigr,r2,l2,p2,sigr2)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2 + r2*exp(-abs(x/l2).^p2)+(x==0).*sigr2^2); % cov function with two time scales
      
rho_r = 1000;  % prior variance of m
lr = 100; % length scale
pr = 1.5; % power  (note: only valid covariance function for p<=2)
sig_r = 30;
r2 = 100;
l2 = 500;
p2 = 1.5;
sigr2 = 30;

kfunplot = kfun(xx,rho_r,lr,pr,sig_r,r2,l2,p2,sigr2);
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');

%% find covariance
% %tune true hyperparameters
prs0 = [rho_r,lr,pr,sig_r]'; % "true" params
LB = [.01,.5, .1, .01]'; % lower bound
UB = [2*var(rr),2*length(rr),2,std(rr)]'; % upper bound
% Set initial params
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB
% set optimization options
opts = optimset('display','iter');

rr_ctr = rr-nanmean(rr);
% anonymous function for estimating kernel covariance
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
% make anonymous function (function pointer) for neg log-likelihood function.
lfun = @(prs)neglogli_GP(prs,kfun,rr_ctr);
% optimize
prsML = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

sigEst_r = prsML(4);
Km = toeplitz(prsML(1)*exp(-abs((1:T)/prsML(2)).^prsML(3)));

%%  hand-tune priors...
rho_b = 2.0;  %var of photon noise
rho_v = 1.5;  %var of motion
rho_f = 2.0;  %var of bleaching
rho_r = sigEst_r^2;  %var of RFP

%% find low-rank covariance matrix
% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = size(X,2);  % number of time points
tt = (1:T)';  % time grid
l = 10; % length scale
Km = kSE(rho_v,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e5;%1e12;  % threshold on condition number
smdiag = diag(Sm); 
% nn = 10;
ii = max(smdiag)./smdiag < thresh;  %logical([ones(1,nn) zeros(1,length(Sm)-nn)]);% vector of indices to keep.
krank = sum(ii); % rank
Ubasis = Um(:,ii);  % basis for Km
Ssqrt = spdiags(sqrt(smdiag(ii)),0,krank,krank); % diagonal matrix sqrt of eigenvalues
Ksqrt = Ubasis*Ssqrt; % low-rank linear operator for generating from iid samples
%%
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues


%% run correction and return parameters...
% [Wt,alpha,tau] = MotionCorrection_rfp_bleach(ff,rr', X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv);
alpha = max(rr);
time = 1:length(rr);
tau = 5000;
B = alpha*exp(-time/tau);
[beta, W, mm] = MotionCorrection_rfp_bleach(rr, B, X, rho_r, Ubasis, Sminv);

%% reconstruct decay and motion
figure
vv = W*X;%sum(Wt.*X);
scal = zeros(1,length(vv));
for ss = 1:length(scal)
    scal(ss) = 1/(1+exp(vv(ss))); 
end
recon = (beta.*scal+mm).*alpha.*exp(-[1:1:length(rr)]/tau);
plot(recon); hold on; plot(rr)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [beta, W, mm] = MotionCorrection_rfp_bleach(rr, B, xx, rho_r, Um, Sminv)
%(ff,rr, X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv)
% MotionCorrection_rfp_bleach - motion correction estimate under iid priors on an
% exponential decay of photobleaching and a scalar modulation of motion
% artifacts (PC1, sum, density)
%
% Computes MAP estimate of aa under the following model:
%    ff = bb + noise_f                 % measured bleaching from immobalized worms
%    rr = (sigmoid(vv)+mm)*bb + noise_r     % measured RFP is the motion-related scaler times photo-bleaching
%                                      % where sigmoid is 1/(1+exp(vv))
% With the following priors:
%    bb ~ N(alph*exp(-t/tau), rho_b*I) % photo-bleaching
%    vv ~ N(W'*X, rho_v*I)             % motion-related fluorescence
%                                      % W is the weights on motion parameter matrix X
%    mm ~ N(0,sigma_m)                 % drawn from a sooth GP to account for additional motion artifact not captured by posture
%
% Inputs:
%    ff [Tx1] - immobalized rfp measurements
%    rr [Tx1] - rfp measurements
%    X  [DxT] - D motion parameters
%  rho_b [1]  - variance of constant photon noise
%  rho_f [1]  - variance of bleaching noise 
%  rho_v [1]  - variance of motion artifacts (residule of the D parameters)
%  rho_r [1]  - variance of RFP 
%
% Output: 
%    Wt [DxT] - estimate of weights on each motion signal (D of them) that
%     vary through time (???)
%  alpha [1]  - amplitude of initial decay
%    tau [1]  - time scale of photo-bleaching

beta0 = 1;
D = size(xx,1);
W0 = ones(1,D);
m0 = rand(1,length(rr));
THETA0 = [beta0, W0, m0];

opts = optimoptions('fmincon', 'display', 'notify');
lb = -inf; % lower bound on W
ub = inf;  % upper bound on W
lfunc = @(THETA) neglogli_motion_bleach2(THETA, rr, D, B, xx, rho_r, Um, Sminv);
temp = fmincon(lfunc,THETA0,[],[],[],[],lb,ub,[],opts);

beta = temp(1);
W = temp(2:2+D-1);
mm = temp(2+D:end);

end

%%%
%prior on w_t series
%time windows(?) to make the problem not ill-posed
%use static w
%%%
%% ---- Loss function ---------
% function negL = neglogli_motion_bleach(THETA, D, fr, xx, Cfr,rho_b, rho_v, tt)
% 
% % Unpack a for likelihood terms
% W = THETA(1:D);
% alpha = THETA(D+1);
% beta = THETA(D+2);
% tau = THETA(D+3);
% vv = sigmoid(W*xx);
% V = [beta*ones(1,length(vv)) ; vv]; 
% CC = Cfr + rho_b*(V*V');
% frctr = fr-V*alpha.*exp(-tt/tau);
% 
% % likelihood terms
% trm_logdet = .5*logdet(CC);  % note logdet should be in your path!
% trm_quad = .5*frctr'*(CC\frctr);
% 
% % prior terms
% trm_logprior = .5/rho_v*(W-0).^2;  %the prior on this modulation is zeros mean (??)
% 
% % Combine terms
% negL = trm_logdet + sum(diag(trm_quad)) + sum(trm_logprior);
% 
% end

%% ---- Logistic function ---------
function S = sigmoid(xx)

S = 1./(1+exp(xx));

end

%% 
% ============ Loss function =====================================
function obj = neglogli_motion_bleach2(THETA, rr, D, B, xx, rho_r, Um, Sminv)
%%%input
% THETA contains the parameters we fit, including scaler beta, weights W, and motion time series m
% rr is the time sereis of RFP signal
% D is the dimension of the motion design matrix xx
% B is the exponential decay of intensity fit from immobalized worms
% rho_r is the std of RFP noise
% The motion m~GP(0,sigma_m), where we approximate the covariance with Um*Sminv*Um'
% 
% Computes MAP estimate of aa under the following model:
%    ff = bb + noise_f                         % measured bleaching from immobalized worms
%    rr = (beta*sigmoid(vv)+mm)*bb + noise_r   % measured RFP is the motion-related scaler times photo-bleaching
%                                              % where sigmoid is 1/(1+exp(vv)) and beta is a scaler
% With the following priors:
%    bb ~ N(alph*exp(-t/tau), rho_b*I) % photo-bleaching
%    vv ~ N(W'*X, rho_v*I)             % motion-related fluorescence
%                                      % W is the weights on motion parameter matrix X
%    mm ~ N(0,sigma_m)                 % drawn from a sooth GP to account for additional motion artifact not captured by posture

beta = THETA(1);
W = THETA(2:2+D-1);
phi = beta*sigmoid(W*xx);
mm = THETA(2+D:end);

% Log-determinant term
dvec = B.^2/rho_r;
M = Sminv + Um'*bsxfun(@times, Um, dvec');
trm_logdet = 0.5*logdet(M) + 0.5*logdet(Sminv) + length(rr)*log(rho_r);  %let C = diag(B)*sigma_m*diag(B)' + rho_r;  this is 0.5*logdet(C)
%not sure if this is correct??

% Quad term
xt = Um'*(rr' - B.*(mm+phi))';
trm_quad = .5*xt'*(M\xt);   %0.5*(r-(phi+m)*B)'*C^-1*(r-(phi+m)*B)

% Prior term
Mm = Sminv + Um'*bsxfun(@times, Um, (mm-zeros(size(mm)))'); %(m-0)'*sigma_m^-1*(m-0)
trm_prior = .5*logdet(Sminv) + .5*logdet(Mm);

% Sum them up
obj = trm_logdet + trm_quad + trm_prior;  %+ trm_diag

end
    