%testscript_simulation_motionbleach
%
close all
clear; clc;
%% Simulation
%ture parameters
T = 300;
time = 1:T;
tau_t = 100;  
alpha_t = 100;
beta_t = 10;
mv = 4;  %smoothness of a motion artifact
W_t = [randn(1,mv-1) rand(1)];
X = [sin(time/pi*0.5)'  0.5*cos(time/pi)'  smooth(randn(1,T),50)  ones(1,length(time))'];  %motion artifact (two frequencies, smooth noise, and an offset)
figure; plot(X); hold on; plot(W_t*X','k')
rho_f = 1;
rho_r = 1;

%bleeching dynamics
B_t = alpha_t*exp(-time./tau_t) + rho_f*randn(1,length(time));  %bleeching baseline
figure; plot(B_t)

%GP of motion-time
% Define anonymous function squared exponential kernel (with strength and length scale)
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
sig_m = .2; % prior standard deviation for movement artifact m
rho_m = sig_m^2;  % prior variance of m
l = 5; % length scale
Km = kSE(rho_m,l,time); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e6;  %1e12;  % threshold on condition number
sdiag = diag(Sm); 
ii = max(sdiag)./sdiag < thresh;  % vector of indices to keep.
krank = sum(ii); % rank
Ubasis = Um(:,ii);  % basis for Km
Ssqrt = spdiags(sqrt(sdiag(ii)),0,krank,krank); % diagonal matrix sqrt of eigenvalues
Ksqrt = Ubasis*Ssqrt; % low-rank linear operator for generating from iid samples
Kapprox = (Ksqrt*Ksqrt'); % low-rank approximation

% Generate movement artifact by sampling from GP
mu_m = 0.;  % mean of movement artifact m
mm = Ksqrt*randn(krank,1) + mu_m; % movement artifact
figure; plot(mm); 
title('True m(t) sampled from GP prior')
xlabel('time (bins)');

%constructing motion-artifact corrupted signal
figure;
phi_t = beta_t*sigmoid(W_t*X');
RR = (phi_t + mm').*B_t + randn(1,length(time))*rho_r;
title('True simulated R(t)')
plot(RR)


%% %%% MLE for decay due to photo-bleeching from immobalized worms%%%
%imobolized GFP
ff = B_t;
fVals = ff;
%exponential form
%%%for single-exponential
% Fexponent = fittype('a*exp(b*x)+c','dependent',{'y'},'independent',...
% {'x'},'coefficients',{'a', 'b', 'c'});
%%%test with doulbe exponent
Fexponent = fittype('a*exp(b*x)+c*exp(d*x)+e','dependent',{'y'},'independent',...
{'x'},'coefficients',{'a', 'b', 'c', 'd', 'e'});
%normalization and removing NaNs
present = (~isnan(ff)');
present = present & (ff~=0)';
xVals = (1:length(ff))';
xVals = xVals(present);
fVals = fVals(present);
denom_f = max(fVals);%(1);  %decaying from the first element
fVals = (fVals/denom_f);  %normalization
%all the fitting options required (following Jeff's hand-tune)
minWindow = 50;
fitOptions = fitoptions(Fexponent);
fitOptions.Lower = [0,-.2,0];
fitOptions.Upper = [1000,0,10000];
%%%for single-exponential
% fitOptions.StartPoint = [range(fVals(fVals~=0)),-.0006,min(fVals(fVals~=0))];
%%%for double-exponentials
fitOptions.StartPoint=[range(fVals(fVals~=0)),-.0006,range(fVals(fVals~=0)),-.0006,min(fVals(fVals~=0))];
fitOptions.Weights = zeros(size(fVals));
fitOptions.Weights(minWindow:end-minWindow) = 1;        
%do exponential fitting
[f_decay,fout] = fit(xVals,fVals',Fexponent,fitOptions);
%now we have the decay form
%%%single-exp
% reconstruct_bleech = f_decay.a*exp(f_decay.b*xVals) + f_decay.c;
%%%double-exp
reconstruct_bleech = f_decay.a*exp(f_decay.b*xVals)+f_decay.c*exp(f_decay.d*xVals)+f_decay.e;
B_rec = (denom_f*reconstruct_bleech)';
figure()
plot(ff)
hold on
plot(B_rec)
title('bleaching dynamics'); xlabel('time');
legend('immobalized signal', 'fit');

%%%learned parameters: alpha and tau via MLE exponential fit
alpha_ = f_decay.a*denom_f;
tau_ = -1/f_decay.b;

%% %%% GP for motion artifact %%%
%%%moving GFP
rr = RR./B_rec;
X = X;
%% GP prior on parameters
%%%%%%%%%%%
%% check covariance for initial condition
figure()
%rr = mm;
%rr = RR;  %real scenario??
%rr = rr-B_t';  %%removing the dacay baseline?
nlags = 100;
xx = -nlags:nlags;
xcsamp = xcov(mm-nanmean(mm),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');

% Define true GP covariance function
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov      
rho_r = 0.03;  % prior variance of m
l_r = 8; % length scale
pr = 2; % power  (note: only valid covariance function for p<=2)
sig_r = 0.05;

% make anonymous function (function pointer) for neg log-likelihood function.
lfun = @(prs)neglogli_GP(prs,kfun,mm);
% Set initial params
LB = [1e-03, 0.5, 0.1, 1e-03]'; % lower bound
UB = [10, 20, 2, 10]';%[1e03, 2*length(rr), 2, 1e03]'; % upper bound
prs0 = [rho_r, l_r, pr, sig_r]'; % initial values
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB
% set optimization options
opts = optimset('display','iter');
% optimize
prsML = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

%kfunplot = kfun(xx,rho_r,lr,pr,sig_r,r2,l2,p2,sigr2);
%kfunplot = kfun(xx,rho_r,l_r,pr,sig_r);
kfunplot = kfun(xx,prsML(1),prsML(2),prsML(3),prsML(4));
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');


%% run correction and return parameters...
% [Wt,alpha,tau] = MotionCorrection_rfp_bleach(ff,rr', X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv);
B_ = B_rec; %alpha_*exp(-time/tau_);
%[rho_, l_, sig_r_, beta_, W_] = GP_step(RR, B_, X', rho_r, l_r, sig_r);
[rho_, l_, sig_r_, beta_, W_] = GP_step(RR, B_, X', prsML(1),prsML(2),prsML(4));

%% %%% MAP for motion time series %%%
figure()
phi_ = beta_*sigmoid(W_*X');  %reconstruction of the multiplicative motion time series
[mm_] = MAP_step(RR, B_, phi_, rho_, l_, sig_r_);
plot(mm_)
hold on
plot(mm)
xlabel('time')
ylabel('motion m(t)')


%% reconstructing rr (?)
%    rr = (beta*sigmoid(vv)+mm)*bb + noise_r
figure()
recon = (phi_+mm_*1).*B_;
plot(recon)
hold on; plot(RR)
   
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%trm_prior = .5*logdet(Sminv) + .5*xp'*(Sminv\xp);
%%%%%%%%
%functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% test minfunctrm_prior = .5*logdet(Sminv) + .5*xp'*(Sminv\xp);

% lfunc_ = @(THETA) (sum((THETA*X'-phi_t).^2));
% opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
% temp=fmincon(lfunc_,rand(1,4),[],[],[],[],-inf,inf,[],opts);

%% Calling to fit for the GP step
function [r_, l_, sig_r_, beta_, W_] = GP_step(rr, B, xx, r0, l0, sig0)

%initial values
% r0 = 1;
% l0 = 1;
% rho0 = 1;
beta0 = 10;
D = size(xx,1);
W0 = ones(1,D);
THETA0 = [r0, l0, sig0, beta0, W0];

%optimization
%opts = optimoptions('fmincon', 'display', 'notify');
opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
lb = [0,0,0,0,-10,-10, -10, -10]; %-inf; %lower bound
ub = ones(1,8)*100;  % inf; %upper bound
lfunc = @(THETA) neglogli_GP2(THETA, rr, B, xx);
temp = fmincon(lfunc,THETA0,[],[],[],[],lb,ub,[],opts);

%return parameter fit
r_ = temp(1);
l_ = temp(2);
sig_r_ = temp(3);
beta_ = temp(4);
W_ = temp(5:5+D-1);

end


%% ----- Logistic function -----
function S = sigmoid(xx)

S = 1./(1+exp(xx));

end

%% ----- negative log-likelihood for GP -----
function obj = neglogli_GP2(THETA, rr, B, xx)
%%%input
% THETA contains the parameters we fit, including r and l for the GP kernel
% parameters, rho_r for the noise strength, and beta to scale the intensity
% and W to weight on the motion variables in the sigmoid fiunction
% rr is the time sereis of RFP signal from the moving worm
% xx is the design matrix with time series of motion variables
% B is the exponential decay of intensity fit from immobalized worms (via exponential or double exponential fitting)

% The motion m~GP(0,sigma_m), where sigma_m is a squared-exponential kerenel governed by r and l, and we approximate the covariance with Um*Sminv*Um'
% Computes marginal likelihood of these parameters under the following model:
%    ff = bb + noise_f                         % measured bleaching from immobalized worms
%    rr = (beta*sigmoid(vv)+mm)*bb + noise_r   % measured RFP is the motion-related scaler times photo-bleaching
%                                              % where sigmoid is 1/(1+exp(vv)) and beta is a scaler
% With the following priors:
%    bb ~ N(alph*exp(-t/tau), rho_b*I) % photo-bleaching (this is fit in the first step already)
%    vv ~ N(W'*X, rho_v*I)             % motion-related fluorescence
%                                      % W is the weights on motion parameter matrix X
%    mm ~ N(0,sigma_m)                 % drawn from a sooth GP to account for additional motion artifact not captured by posture

%unpack parameters
D = size(xx,1);  %dimension of the motion variables
r = THETA(1);  %rho stength of the covriance
l = THETA(2);  %length scale of the Gaussian kernel
sig = THETA(3);  %noise strength at zero time lag
beta = THETA(4);  %rescale factor compare to immobalize worms
W = THETA(5:5+D-1);  %weights on the motion artifacts
phi = beta*sigmoid(W*xx);  %reconstruction of the motion artifact via design matrix and sigmoid function

% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = size(xx,2);  % number of time points
tt = (1:T)';  % time grid
Km = kSE(r,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e10;%1e12;  % threshold on condition number
smdiag = diag(Sm); 
ii = max(smdiag)./smdiag < thresh;  %logical([ones(1,nn) zeros(1,length(Sm)-nn)]);% vector of indices to keep.
krank = sum(ii); % rank
Um = Um(:,ii);  % basis for Km
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues

% Log-determinant term
dvec = B.^2/sig^2;
M = Sminv + Um'*bsxfun(@times, Um, dvec');
trm_logdet = 0.5*logdet(M) + 0.5*length(rr)*log(sig) + 0.5*logdet(Sminv);  %let C = diag(B)*sigma_m*diag(B)' + rho_r;  this is 0.5*logdet(C)
%not sure if this is correct??

% Diag term
trm_diag = .5*(sum((rr - B.*phi).^2))/sig^2;

% Quad term
xt = Um'*((rr - B.*phi).*(B/sig^2))';
trm_quad = .5*xt'*(M\xt);   %0.5*(r-(phi+m)*B)'*C^-1*(r-(phi+m)*B) with mean(m)=0

% Sum them up
obj = trm_logdet + trm_diag + trm_quad;

end

%% Calling to fit for them MAP step
function [mm] = MAP_step(rr, B, phi, r, l, sig)

%initial values
THETA_m = randn(1,length(rr));

% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = length(rr);  % number of time points
tt = (1:T)';  % time grid
Km = kSE(r,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e10;%1e12;  % threshold on condition number
smdiag = diag(Sm); 
ii = max(smdiag)./smdiag < thresh;  %logical([ones(1,nn) zeros(1,length(Sm)-nn)]);% vector of indices to keep.
krank = sum(ii); % rank
Um = Um(:,ii);  % basis for Km
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues

%optimization
%opts = optimoptions('fmincon', 'display', 'notify');
opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
lb = -1000; %-inf; % lower bound
ub = 1000;%inf;  % upper bound
lfunc = @(mm) neglogli_MAP_mm(mm, rr, B, phi, sig, Um, Sminv);
temp = fmincon(lfunc,THETA_m,[],[],[],[],lb,ub,[],opts);

%return parameter fit
mm = temp;

end
%% ----- MAP estimate for motion through time -----
function obj = neglogli_MAP_mm(mm, rr, B, phi, sig, Um, Sminv)
%MAP estimate with m~GP(0,K)
%where K is parameterized as Gaussian kernels with covariance structure
%described by squared exponential kernels and with low-dimension
%approximation: Um*Sminv*Um'
%Other variables fit from MLE and GP are B decay, rr recoding, and phi
%motion variable

% Log-determinant term
dvec = B.^2/sig^2;
M = Sminv + Um'*bsxfun(@times, Um, dvec');
trm_logdet = 0.5*logdet(M);% + 0.5*logdet(Sminv) + length(rr)*log(rho_r);  %let C = diag(B)*sigma_m*diag(B)' + rho_r;  this is 0.5*logdet(C)
%not sure if this is correct??

% Quad term
% xt = Um'*(rr - B.*(mm+phi))';
xt = Um'*((rr - B.*(phi+mm)).*(B/sig^2))';
trm_quad = .5*xt'*(M\xt);   %0.5*(r-(phi+m)*B)'*C^-1*(r-(phi+m)*B)

% Diag term
%trm_diag = .5*(sum((rr - B.*phi).^2))/sig^2;

% Prior term
%Mm = Sminv + Um'*bsxfun(@times, Um, (mm-zeros(size(mm)))'); %(m-0)'*sigma_m^-1*(m-0)
xp = Um'*(mm-zeros(size(mm)))';
trm_prior = .5*logdet(Sminv) + .5*xp'*(Sminv\xp);
%trm_prior = .5*logdet(Mm);

% Sum them up
obj = trm_quad + trm_prior;  %+ trm_diag + trm_logdet + 

end
    
