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
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
      
rho_r = 1000;  % prior variance of m
lr = 100; % length scale
pr = 1.5; % power  (note: only valid covariance function for p<=2)
sig_r = 30;

kfunplot = kfun(xx,rho_r,lr,pr,sig_r);
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
rho_r = sigEst_r^2  %var of RFP

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
nn = 10;
ii = logical([ones(1,nn) zeros(1,length(Sm)-nn)]);  %max(smdiag)./smdiag < thresh;  % vector of indices to keep.
krank = sum(ii); % rank
Ubasis = Um(:,ii);  % basis for Km
Ssqrt = spdiags(sqrt(smdiag(ii)),0,krank,krank); % diagonal matrix sqrt of eigenvalues
Ksqrt = Ubasis*Ssqrt; % low-rank linear operator for generating from iid samples
%%
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues


%% run correction and return parameters...
[Wt,alpha,tau] = MotionCorrection_rfp_bleach(ff,rr', X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv);


%% reconstruct decay and motion
figure
vv = Wt*X;%sum(Wt.*X);
scal = zeros(1,length(vv));
for ss = 1:length(scal)
    scal(ss) = 1/(1+exp(vv(ss))); 
end
recon = scal.*alpha.*exp(-[1:1:length(rr)]/tau);
plot(recon); hold on; plot(rr)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Wt,alpha,tau] = MotionCorrection_rfp_bleach(ff,rr, X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv)
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

T = length(rr); % number of timesteps
Cfr = diag([rho_f, rho_r]); % covariance for measurement noise

%%%initial values (??)
D = size(X,1);  %D motion signals
W0 = ones(D,1);
alpha0 = 300;  %amplitude of intial condition of 0:imm and 1:mov
beta0 = 1;  %ratio between immobalized and moving amplitude
tau0 = 1000;  %decay exponent
THETA0 = [W0', [alpha0,beta0], tau0];

opts = optimoptions('fmincon', 'display', 'notify');
lb = -inf; % lower bound on W
ub = inf;  % upper bound on W

% Optimize for each time bin 
% (could parallelize with parfor, since independent for each bin!)
% Wt = zeros(D,T);
% fprintf('Motion&BleachCorrection: estimating time bin\n ');
% for jj = 1:T
%     if mod(jj,10)==0
%         fprintf('%\n ',jj);
%     end
%   % set loss function
%     lfunc = @(THETA) neglogli_motion_bleach(THETA, D,[ff(jj);rr(jj)], X(:,jj), Cfr, rho_b, rho_v, jj);
%     temp = fmincon(lfunc,THETA0,[],[],[],[],lb,ub,[],opts);  %returning estmate of W at time t, alpha, and tau
%     Wt(:,jj) = temp(1:D);  %just storing weights for now
% %     fprint(temp(end-1:end)) %check stationarity (??)
% end

time_ = 1:T;
% lfunc = @(THETA) neglogli_motion_bleach(THETA, D,[ff;rr], X, Cfr, rho_b, rho_v, time_);
lfunc = @(THETA) neglogli_motion_bleach2(THETA, D,ff,rr, X,rho_b, rho_v,rho_f,rho_r, Ubasis, Sminv);
temp = fmincon(lfunc,THETA0,[],[],[],[],lb,ub,[],opts);

Wt = temp(1:D);
alpha = temp(D+1);
% beta = temp(D+2);
tau = temp(D+2);
fprintf('\n');
end

%%%
%prior on w_t series
%time windows(?) to make the problem not ill-posed
%use static w
%%%
%% ---- Loss function ---------
function negL = neglogli_motion_bleach(THETA, D, fr, xx, Cfr,rho_b, rho_v, tt)

% Unpack a for likelihood terms
W = THETA(1:D);
alpha = THETA(D+1);
beta = THETA(D+2);
tau = THETA(D+3);
vv = sigmoid(W*xx);
V = [beta*ones(1,length(vv)) ; vv]; 
CC = Cfr + rho_b*(V*V');
frctr = fr-V*alpha.*exp(-tt/tau);

% likelihood terms
trm_logdet = .5*logdet(CC);  % note logdet should be in your path!
trm_quad = .5*frctr'*(CC\frctr);

% prior terms
trm_logprior = .5/rho_v*(W-0).^2;  %the prior on this modulation is zeros mean (??)

% Combine terms
negL = trm_logdet + sum(diag(trm_quad)) + sum(trm_logprior);

end

%% ---- Logistic function ---------
function S = sigmoid(xx)

S = 1./(1+exp(xx));

end

%% 
% ============ Loss function =====================================
function obj = neglogli_motion_bleach2(THETA, D, ff, rr, xx, rho_b, rho_v,rho_f,rho_r, Um, Sminv)
%(aa,rr,gg,rho_r,rho_g,mu_a,rho_a,mu_m,Um,Sminv)

% parameters to fit
W = THETA(1:D);
alpha = THETA(D+1);
beta = 1;  %THETA(D+2);
tau = THETA(D+2);

% Log-determinant term
aa = sigmoid(W*xx);
dvec = (1/rho_f + 1./rho_r*aa.^2);%/rho_b^2;
M = (Sminv + Um'*bsxfun(@times, Um, dvec'));%*rho_b^2;
trm_logdet = .5*logdet(M);
% % M = Sminv + Um'*bsxfun(@times, Um, dvec');  %main part to modify for motion smooth GP prior adding to motion
%%%%% GP in motion test %%%%%
%%det(M)=det(A−BD^-1C)det(D)
% A = rho_b^2*diag(ones(1,size(Sminv,1))*beta^2) + rho_f^2;
% BDC = beta^2*Sminv;
% M = A+BDC;
% logdetD = prod(diag(Sminv)+rho_r);
% trm_logdet = 0.5*logdet(M) + 0.5*logdetD;
% logSig = Um*Sminv*Um';  %prior term on motion covariance
% V = [diag(ones(1,length(ff))*beta^2); logSig];  %2T x T
% tempM = V*V'*rho_b^2;
% D = diag([1/rho_f*ones(1,length(ff))  1/rho_r*aa.^2]);
% M = tempM + D;  %not sure if this is correct: [beta^2  beta*sigma;  beta*sigma  sigma^2] block matrix
% trm_logdet = .5*logdet(M) + 0.5*prod(diag(prod(Sminv)));  %real(logdet(logSig));

% Diag term
tt = 1:length(ff);
mu_m = alpha.*exp(-tt/tau);%%%(???)
trm_diag = .5*(sum((ff-mu_m).^2)/rho_f + sum((rr-mu_m.*aa).^2)./rho_r);

% Quad term
xt = Um'*((ff-mu_m)/rho_f + aa.*(rr-mu_m.*aa)/rho_r)';
% xv = [ff rr];
% xt = (xv - [beta*ones(1,length(ff)).*mu_m aa.*mu_m])';
trm_quad = -.5*xt'*(M\xt);

% Prior term
trm_prior = .5/rho_v*sum((W*xx-0).^2);  %not sure if this is correct

% Sum them up
obj = trm_logdet + trm_quad + trm_prior + trm_diag; 
%obj(isnan(obj)) = inf;  %%%debugging
    
end
    