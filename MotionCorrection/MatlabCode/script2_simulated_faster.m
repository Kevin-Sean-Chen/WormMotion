% Script for simulating data for motion artifact problem and estimating neural
% activity using smooth GP prior on m

%% 1. Generate movement artifact from GP

% Define anonymous function squared exponential kernel
kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));

T = 300;  % number of time points
tt = (1:T)';  % time grid
sig_m = .2; % prior standard deviation for movement artifact m
rho_m = sig_m^2;  % prior variance of m
l = 10; % length scale
Km = kSE(rho_m,l,tt); % the T x T GP covariance

% Find low-rank approximation to Km using SVD
[Um,Sm] = svd(Km);
thresh = 1e12;  % threshold on condition number
smdiag = diag(Sm); 
ii = max(smdiag)./smdiag < thresh;  % vector of indices to keep.
krank = sum(ii); % rank
Ubasis = Um(:,ii);  % basis for Km
Ssqrt = spdiags(sqrt(smdiag(ii)),0,krank,krank); % diagonal matrix sqrt of eigenvalues
Ksqrt = Ubasis*Ssqrt; % low-rank linear operator for generating from iid samples

% Generate movement artifact by sampling from GP
mu_m = 1;  % mean of movement artifact m
mm = Ksqrt*randn(krank,1) + mu_m; % movement artifact

% Plot it
subplot(111); plot(tt,mm); 
title('True m(t) sampled from GP prior')
xlabel('time (bins)');

%% 2. Generate neural activity signals y and a, and measured signals r and g

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

%% 2. Estimate a(t) with iid priors on m(t) and a(t) (JP code)

aa1 = MotionCorrection_iid(rr,gg,sig_r^2,sig_g^2,mu_m,rho_m,mu_a,rho_a);
yy1 = D*aa1; % corresponding estimate of y(t)

%% 3. find neural activity y(t), smooth prior (Lea code)

yy0 = D*aa0; % initial estimate of y(t) (from r/g estimate).
tic;
yy2 = MotionCorrection(rr,gg,yy0,sig_r^2,sig_g^2,mu_y,rho_y,mu_m.*ones(T,1),Ubasis,Ssqrt.^2,D);
toc;
aa2 = D\yy2; % corresponding estimate of a(t)

%% 4. find fluorescence a(t), smooth prior (JP code).

Sminv = spdiags(1./smdiag(ii),0,krank,krank); % diagonal matrix sqrt of eigenvalues
aa3 = MotionCorrection_smooth_aa(rr,gg,aa0,sig_r^2,sig_g^2,mu_a,rho_a,mu_m,Ubasis,Sminv,D);
yy3 = D*aa3;  % corresponding estimate of y(t)

%% 4b. find neural activity y(t), smooth prior (JP code).

tic;
yy3b = MotionCorrection_smooth_yy(rr,gg,aa0,sig_r^2,sig_g^2,mu_y,rho_y,mu_m,Ubasis,Sminv,D);
toc;
aa3b = D\yy3b;  % corresponding estimate of y(t)


%% Plot results

subplot(211);  % a(t) plots
plot(tt, [aa aa0 aa1 aa2 aa3 aa3b]); 
title('True and estimated a(t)');
legend('true aa', 'r/g', 'iid prior', 'smooth y', 'smooth a');
  
subplot(212) % y(t) plots
plot(tt,mu_y,'--k',tt,[yy yy1 yy2 yy3 yy3b]);
title('True and estimated y(t)');
legend('prior mean', 'true', 'iid', 'smooth y', 'smooth a');

% Report errors
amse = @(x)(norm(x-aa)^2);
ymse = @(x)(norm(x-yy)^2);
fprintf('=====\nErrs in a(t):\n r/g: %9.2f\n iid: %9.2f\n smth y: %6.2f (lea)\n smth a: %6.2f\n smth y2: %5.2f (jp)\n', ...
    amse(aa0), amse(aa1), amse(aa2), amse(aa3), amse(aa3b));
fprintf('-----\nErrs in y(t):\n r/g: %9.2f\n iid: %9.2f\n smth y: %6.2f (lea)\n smth a: %6.2f\n smth y2: %5.2f (jp)\n', ...
    ymse(yy0), ymse(yy1), ymse(yy2), ymse(yy3), ymse(yy3b));
