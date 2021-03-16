%testscript_RFP_motionbleach
%
%% load file
%imobolized GFP
% file = '/tigress/LEIFER/PanNeuronal/2018/20180518/BrainScanner20180518_093125/'; %BrainScanner20180518_091402 %BrainScanner20180518_094052
% file_imm = '/tigress/LEIFER/PanNeuronal/2017/20171128/BrainScanner20171128_154753/'; %%%very immobalized GFP worm
% load([file_imm,'heatData.mat'])
% FF = rRaw;

%%%moving GFP
% file_mov = '/tigress/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_155051/';%BrainScanner20160506_160928/';%';%';  %%%moving GFP worms
% load([file_mov,'heatData.mat'])
% RR = rRaw;
% GG = gRaw;

% %%%moving GCaMP
% file = '/tigress/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620/';
% load([file_mov,'heatData.mat'])
% RR = rRaw;
% GG = gRaw;

%% %%% MLE for decay due to photo-bleeching from immobalized worms%%%
%imobolized GFP
file = '/tigress/LEIFER/PanNeuronal/2018/20180518/BrainScanner20180518_093125/'; %BrainScanner20180518_091402 %BrainScanner20180518_094052
file_imm = '/tigress/LEIFER/PanNeuronal/2017/20171128/BrainScanner20171128_154753/'; %%%very immobalized GFP worm
load([file_imm,'heatData.mat'])
FF = rRaw;
GG = gRaw;

%make data nicer (remove weird data points..., NaN and blow-ups)
remove = 100;
lim = 3000;%min(size(FF,2));%,size(RR,2));
nFF = FF(:,remove:lim);
nFF(nFF>2000) = nanmean(nanmean(nFF));
nFF(isnan(nFF)) = nanmean(nanmean(nFF));  %roughly remove them for now (??)
% subplot(121); imagesc(nFF); subplot(122); imagesc(nRR);

%call for a cell
cellID = 10;
ff = smooth(nFF(cellID,:),1);
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
minWindow = 150;
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
[f_decay,fout] = fit(xVals,fVals,Fexponent,fitOptions);
%now we have the decay form
%%%single-exp
% reconstruct_bleech = f_decay.a*exp(f_decay.b*xVals) + f_decay.c;
%%%double-exp
reconstruct_bleech = f_decay.a*exp(f_decay.b*xVals)+f_decay.c*exp(f_decay.d*xVals)+f_decay.e
B = denom_f*reconstruct_bleech;
plot(ff)
hold on
plot(B)
title('bleaching dynamics'); xlabel('time');
legend('immobalized signal', 'fit');

%%%learned parameters: alpha and tau via MLE exponential fit
alpha = f_decay.a;
tau = -1/f_decay.b;

%% %%% GP for motion artifact %%%
%%%moving GFP
file_mov = '/tigress/LEIFER/PanNeuronal/2016/20160506/BrainScanner20160506_155051/';%BrainScanner20160506_160928/';%';%';  %%%moving GFP worms
load([file_mov,'heatData.mat'])
RR = rRaw;
GG = gRaw;

%make data nicer (remove weird data points..., NaN and blow-ups)
remove = 100;
% lim = 500;%min(size(RR,2),size(RR,2));
nRR = RR(:,remove:lim);
nRR(nRR>2000) = nanmean(nanmean(nRR));
nRR(isnan(nRR)) = nanmean(nanmean(nRR));
% subplot(121); imagesc(nFF); subplot(122); imagesc(nRR);

rr = nRR(cellID,:);
%load behavioral variables
load([file_mov,'heatDataMS.mat']);
PC1 = behavior.pc1_2(:,1);
PC2 = behavior.pc1_2(:,2);
%%%% Other factors: local density, CMOS space, and maybe scanning artifact
%global change
fluo = nanmean(RR,1);
%design matrix
% remove = 1; %can remove some initial artifacts in the future
% X = [ones(length(rr),1)  PC1(remove:end) fluo(remove:end)']';
X = [ones(length(rr),1)  PC1(remove:lim) fluo(remove:lim)']';
X(isnan(X)) = 0;
%% GP prior on parameters
%%%%%%%%%%%
%% check covariance for initial condition
nlags = 300;
xx = -nlags:nlags;
% rr = rr-B';  %%removing the dacay baseline?
xcsamp = xcov(rr-nanmean(rr),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');

% Define true GP covariance function
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov
% kfun = @(x,r,l,p,sigr,r2,l2,p2,sigr2)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2 + r2*exp(-abs(x/l2).^p2)+(x==0).*sigr2^2); % cov function with two time scales
      
rho_r = 1000;  % prior variance of m
lr = 100; % length scale
pr = 2; % power  (note: only valid covariance function for p<=2)
sig_r = 100;
% r2 = 100;
% l2 = 500;
% p2 = 1.5;
% sigr2 = 30;

%kfunplot = kfun(xx,rho_r,lr,pr,sig_r,r2,l2,p2,sigr2);
kfunplot = kfun(xx,rho_r,lr,pr,sig_r);
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');


%% run correction and return parameters...
% [Wt,alpha,tau] = MotionCorrection_rfp_bleach(ff,rr', X, rho_b,rho_v,rho_f,rho_r, Ubasis, Sminv);
time = 1:length(rr);
B = alpha*exp(-time/tau);
[r_, l_, rho_r_, beta_, W_] = GP_step(rr, B, X, rho_r, lr, sig_r);

%% test to reconstruct decay and motion
% figure
% vv = W*X;%sum(Wt.*X);
% scal = zeros(1,length(vv));
% for ss = 1:length(scal)
%     scal(ss) = 1/(1+exp(vv(ss))); 
% end
% recon = (beta.*scal+mm).*alpha.*exp(-[1:1:length(rr)]/tau);
% plot(recon); hold on; plot(rr)

%% %%% MAP for motion time series %%%
figure()
phi = beta_*sigmoid(W_*X);  %reconstruction of the multiplicative motion time series
[mm] = MAP_step(rr, B, phi, r_, l_, rho_r_);
plot(mm)
xlabel('time')
ylabel('motion m(t)')

%% reconstructing rr (?)
%    rr = (beta*sigmoid(vv)+mm)*bb + noise_r
figure()
plot((phi+mm*0.05).*B')
   
%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calling to fit for the GP step
function [r_, l_, rho_r_, beta_, W_] = GP_step(rr, B, xx, r0, l0, rho0)

%initial values
% r0 = 1;
% l0 = 1;
% rho0 = 1;
beta0 = 1;
D = size(xx,1);
W0 = ones(1,D);
THETA0 = [r0, l0, rho0, beta0, W0];

%optimization
opts = optimoptions('fmincon', 'display', 'notify');
lb = -inf; % lower bound
ub = inf;  % upper bound
lfunc = @(THETA) neglogli_GP(THETA, rr, B, xx);
temp = fmincon(lfunc,THETA0,[],[],[],[],lb,ub,[],opts);

%return parameter fit
r_ = temp(1);
l_ = temp(2);
rho_r_ = temp(3);
beta_ = temp(4);
W_ = temp(5:5+D-1);

end


%% ----- Logistic function -----
function S = sigmoid(xx)

S = 1./(1+exp(xx));

end

%% ----- negative log-likelihood for GP -----
function obj = neglogli_GP(THETA, rr, B, xx)
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
r = THETA(1);
l = THETA(2);
rho_r = THETA(3);
beta = THETA(4);
W = THETA(5:5+D-1);
phi = beta*sigmoid(W*xx);

% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = size(xx,2);  % number of time points
tt = (1:T)';  % time grid
Km = kSE(r,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e6;%1e12;  % threshold on condition number
smdiag = diag(Sm); 
ii = max(smdiag)./smdiag < thresh;  %logical([ones(1,nn) zeros(1,length(Sm)-nn)]);% vector of indices to keep.
krank = sum(ii); % rank
Um = Um(:,ii);  % basis for Km
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues

% Log-determinant term
dvec = B.^2/rho_r;
M = Sminv + Um'*bsxfun(@times, Um, dvec');
trm_logdet = 0.5*logdet(M) + 0.5*logdet(Sminv) + length(rr)*log(rho_r);  %let C = diag(B)*sigma_m*diag(B)' + rho_r;  this is 0.5*logdet(C)
%not sure if this is correct??

% Quad term
xt = Um'*(rr - B.*phi)';
trm_quad = .5*xt'*(M\xt);   %0.5*(r-(phi+m)*B)'*C^-1*(r-(phi+m)*B) with mean(m)=0

% Sum them up
obj = trm_logdet + trm_quad;

end

%% Calling to fit for them MAP step
function [mm] = MAP_step(rr, B, phi, r, l, rho_r)

%initial values
THETA_m = randn(1,length(rr));

% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = length(rr);  % number of time points
tt = (1:T)';  % time grid
Km = kSE(r,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e6;%1e12;  % threshold on condition number
smdiag = diag(Sm); 
ii = max(smdiag)./smdiag < thresh;  %logical([ones(1,nn) zeros(1,length(Sm)-nn)]);% vector of indices to keep.
krank = sum(ii); % rank
Um = Um(:,ii);  % basis for Km
Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues

%optimization
opts = optimoptions('fmincon', 'display', 'notify');
lb = -inf; % lower bound
ub = inf;  % upper bound
lfunc = @(mm) neglogli_MAP_mm(mm, rr, B, phi, rho_r, Um, Sminv);
temp = fmincon(lfunc,THETA_m,[],[],[],[],lb,ub,[],opts);

%return parameter fit
mm = temp;

end
%% ----- MAP estimate for motion through time -----
function obj = neglogli_MAP_mm(mm, rr, B, phi, rho_r, Um, Sminv)
%MAP estimate with m~GP(0,K)
%where K is parameterized as Gaussian kernels with covariance structure
%described by squared exponential kernels and with low-dimension
%approximation: Um*Sminv*Um'
%Other variables fit from MLE and GP are B decay, rr recoding, and phi
%motion variable

% Log-determinant term
dvec = B.^2/rho_r;
M = Sminv + Um'*bsxfun(@times, Um, dvec');
trm_logdet = 0.5*logdet(M) + 0.5*logdet(Sminv) + length(rr)*log(rho_r);  %let C = diag(B)*sigma_m*diag(B)' + rho_r;  this is 0.5*logdet(C)
%not sure if this is correct??

% Quad term
xt = Um'*(rr - B.*(mm+phi))';
trm_quad = .5*xt'*(M\xt);   %0.5*(r-(phi+m)*B)'*C^-1*(r-(phi+m)*B)

% Prior term
% Mm = Sminv + Um'*bsxfun(@times, Um, (mm-zeros(size(mm)))'); %(m-0)'*sigma_m^-1*(m-0)
xp = Um'*(mm-zeros(size(mm)))';
trm_prior = .5*logdet(Sminv) + .5*xp'*(Sminv\xp);
% trm_prior = .5*logdet(Sminv) + .5*logdet(Mm);

% Sum them up
obj = trm_logdet + trm_quad + trm_prior;  %+ trm_diag

end
    
