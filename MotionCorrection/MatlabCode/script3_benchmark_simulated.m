% test script to fit all parameters and extract neural activity from 
% simulated data to see if the overall fitting procedure works
%% 1. make some simulated data:
%% 1.1 Generate movement artifact from GP
% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
T = 300;  % number of time points
tt = (1:T)';  % time grid
sig_m = .2; % prior standard deviation for movement artifact m
rho_m = sig_m^2;  % prior variance of m
lm = 10; % length scale
Km = kSE(rho_m,lm,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e12;  % threshold on condition number
smdiag = diag(Sm); subplot(221); imagesc(Km); axis image; title('GP prior covariance');
subplot(222); imagesc(Kapprox); axis image; title('low-rank approximation');
subplot(223); plot(tt,Km(:,1:50:T),'b',tt,Kapprox(:,1:50:T),'r--');
title('slices');
ii = max(smdiag)./smdiag < thresh;  % vector of indices to keep.
krank = sum(ii); % rank
Ubasis = Um(:,ii);  % basis for Kmsubplot(221); imagesc(Km); axis image; title('GP prior covariance');
subplot(222); imagesc(Kapprox); axis image; title('low-rank approximation');
subplot(223); plot(tt,Km(:,1:50:T),'b',tt,Kapprox(:,1:50:T),'r--');
title('slices');
Ssqrt = spdiags(sqrt(smdiag(ii)),0,krank,krank); % diagonal matrix sqrt of eigenvalues
Ksqrt = Ubasis*Ssqrt; % low-rank linear operator for generating from iid samples

% Generate movement artifact by sampling from GP
mu_m = 1;  % mean of movement artifact m
mm = Ksqrt*randn(krank,1) + mu_m; % movement artifact

% Plot it
subplot(111); plot(tt,mm); 
title('True m(t) sampled from GP prior')
xlabel('time (bins)');

%% 1.2 Generate neural activity signals y and a, and measured signals r and g

% Set params
sig_r = .25; % prior stdev of r noise
sig_g = .25; % prior stdev of g noise
alpha_g = 0.9; % single time-bin decay of gcamp fluorescence signal
tau_g = -1/log(alpha_g);  % time constant of gcamp decay (in time bins) 

% Generate neural activity yy from GP

mu_y = 1*ones(T,1); % mean of neural activity y
rho_y = 0.1;  % variance of true y
l_y = 2;      % length scale of true y
Ky = kSE(rho_y,l_y,tt); % true covariance kernel
[Uy,Sy] = svd(Ky);
yy = Uy*sqrt(Sy)*randn(T,1)+mu_y; % generate noise

% Generate measured activity-related fluorescence aa
D = spdiags(ones(T,1)*[-alpha_g 1],-1:0,T,T); % better implementation!
aa = D\yy; % neural activity-related fluorescence signal a(t)

% Compute params of prior over aa
mu_a = mean(D\mu_y); % mean of a(t)
rho_a = var(aa);  % variance of a(t) 

% Generate R and G
rr = mm + sig_r*randn(T,1);
gg = mm.*aa + sig_g*randn(T,1);

% Make plots
subplot(311);
plot(tt,mm,tt,rr); 
title('motion artifact & measured rfp'); legend('artifact', 'rfp');

subplot(312);
plot(tt,gg);  title('measured gcamp');

subplot(313);
aa0 = gg./rr;
plot(tt,aa,tt, aa0);
legend('true aa', 'g/r estimate');
title('activity-related fluorescence')

prs_true = [rho_m,lm,mu_m,2, sig_r,sig_g]; % save true parameter values
%% 2. find ML estimates of the GP prior over motion, variances for r and g
prs0 = [1 1 2 std(rr)]'; % initial values for [rho_m,lm,pm,sig_r]
LB = [1e-03,.5, .1, 1e-03]'; % lower bound
UB = [1e03,2*length(rr),2,1e03]'; % upper bound

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
sigEst_y = 1;


% compute estimates of full covariances etc needed later in motion
% correction

Sminv = spdiags(1./smdiag(ii),0,krank,krank); 

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


% compare results to true data
fprintf('\t ============================================\n')
fprintf('\t\t true \t\t estimate: \t MSE:\n')
fprintf('\t rho_m:\t %1.4f \t  %1.4f \t %1.4d \n',rho_m,prsML(1),(rho_m-prsML(1)).^2)
fprintf('\t lm_m:\t %1.4f \t %1.4f \t %1.4d \n',lm,prsML(2),(lm-prsML(2)).^2)
fprintf('\t p_m:\t %1.4f \t  %1.4f \t %1.4d \n',2.0,prsML(3),(2.0-prsML(3)).^2)
fprintf('\t mu_m:\t %1.4f \t %1.4f \t %1.4d \n',mu_m,muEst_m,(mu_m-muEst_m).^2)
fprintf('\t sig_r:\t %1.4f \t  %1.4f \t %1.4d \n',sig_r,prsML(4),(sig_r-sigEst_r).^2)
fprintf('\t sig_g:\t %1.4f \t  %1.4f \t %1.4d \n',sig_g,sigEst_g,(sig_g-sigEst_g).^2)
fprintf('\t mu_y: \t %1.4f \t  %1.4f \t %1.4d \n',mean(mu_y),mean(muEst_y),mean((mu_y-muEst_y).^2))
fprintf('\t ============================================\n')
%% 3. Do motion correction
aa0 = gg./rr;
yy0 = D*aa0;

% initialize at true parameter values
yyEst_true = MotionCorrection_smooth_yy(rr,gg,aa0,sig_r^2,sig_g^2,mu_y,rho_y,mu_m,Ubasis,Sminv,D);
aaEst_true = D\yyEst_true;  % corresponding estimate of y(t)

yyEst_est = MotionCorrection_smooth_yy(rr,gg,aa0,sigEst_r^2,sigEst_g^2,muEst_y,sigEst_y,muEst_m,UbasisEst,SminvEst,D);
aaEst_est = D\yyEst_est;  % corresponding estimate of y(t)
%% benchmark method against g/r

subplot(211);  % a(t) plots
plot(tt, [aa aa0 aaEst_true aaEst_est]); 
title('True and estimated a(t)');
legend('true aa', 'g/r', 'supply true prior params', 'supply estimated params');
  
subplot(212) % y(t) plots
plot(tt,mu_y,'--k',tt,[yy yyEst_true yyEst_est]);
title('True and estimated y(t)');
legend('prior mean', 'true', 'supply true prior params', 'supply estimated params');

% Report errors
amse = @(x)(norm(x-aa)^2);
ymse = @(x)(norm(x-yy)^2);
fprintf('=====\nErrs in a(t):\n g/r: %9.2f\n init true: %9.2f\n init est: %6.2f \n', ...
    amse(aa0), amse(aaEst_true), amse(aaEst_est));
fprintf('-----\nErrs in y(t):\n g/r: %9.2f\n init true: %9.2f\n init est: %6.2f \n', ...
    ymse(yy0), ymse(yyEst_true), ymse(yyEst_est));
