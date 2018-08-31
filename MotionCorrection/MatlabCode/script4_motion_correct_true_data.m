% script to look at real GFP data to see if activity is inferred or not
clear all; close all
%% select single neuron from data

load('~/Documents/git/CElegans/SharedData/MATLAB_Files/WormGFP.mat')

nrn  = 20; % just pick an index
T = 300; % set shorter time series
k = 700;
rr = R_Raw(nrn,1+k:T+k)';
gg = G_Raw(nrn,1+k:T+k)';

% standardize rr and gg
rr = rr./std(rr);  %
gg = gg./std(gg);  %

figure;
subplot(211); plot(rr); title('measured motion artifact and rfp')
subplot(212); plot(gg); title('measured gcamp')

alpha_g = 0.7; % single time-bin decay of gcamp fluorescence signal
tau_g = -1/log(alpha_g);  % time constant of gcamp decay (in time bins) 
D = spdiags(ones(T,1)*[-alpha_g 1],-1:0,T,T); % better implementation!
%% Hand-tune the GP params (bye eye)
% Compute autocovariance 
nlags = 300;
xx = -nlags:nlags;
xcsamp = xcov(rr-nanmean(rr),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');

% Define true GP covariance function
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
      
rho = 0.1;  % prior variance of m
lm = 8; % length scale
pm = 1.5; % power  (note: only valid covariance function for p<=2)
sigr = 1;

kfunplot = kfun(xx,rho,lm,pm,sigr);
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');
%% 2. find ML estimates of the GP prior over motion, variances for r and g
prs0 = [rho,lm,pm,sigr]'; % true params
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
aa0 = gg./rr;
yy0 = D*aa0;
sigEst_y = 1;
% sigEst_g = 10;
% sigEst_r = 10;
yyEst = MotionCorrection_smooth_yy(rr,gg,aa0,sigEst_r^2,sigEst_g^2,muEst_y,sigEst_y,muEst_m,UbasisEst,SminvEst,D);
aaEst = D\yyEst;  % corresponding estimate of y(t)

%% evaluate estimates vs g/r

subplot(211);  % a(t) plots
plot(1:T,gg);
subplot(212); 
plot(1:T, [aa0 aaEst]); 
title('estimated a(t)');
legend('g/r', ' motion corrected');
  