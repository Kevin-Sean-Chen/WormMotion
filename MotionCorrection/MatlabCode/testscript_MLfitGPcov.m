% Simple script to illustrate efficient sampling from GP using the
% convolution theorem (i.e., in the fourier domain).

% Define true GP covariance function
rho = 5;  % prior variance of m
lm = 8; % length scale
pm = 1.1; % power  (note: only valid covariance function for p<=2)
sigr = 2;
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov

% Define mean
mu = 10;

% Plot it 
%nlags = ceil(lm*4*.1)*10;  % number of lags for plotting purposes
nlags = 50;
xx = -nlags:nlags;
kfunplot = kfun(xx,rho,lm,pm,sigr);
subplot(211);
plot(xx,kfunplot);
title('true covariance'); xlabel('lag');

%% Make correlated noise with this covariance function
T = 500;  % make sample

% Evaluate covariance function and take FFT
tvec = ([0:floor((T-1)/2),-ceil((T-1)/2):-1])';  % vector of time bins of length T
kf = (kfun(tvec,rho,lm,pm,sigr)); % covariance function of appropriate length
kfh = real(fft(kf)); % take fourier transform

% Generate sample
mt = real(ifft(sqrt(kfh).*fft(randn(T,1))))+mu; 

% Let's check its marginal statistics:
fprintf('mu: %.2f   sample mu: %.2f\n', mu, mean(mt));
fprintf('var: %.2f   sample var: %.2f\n', rho+sigr.^2, var(mt));

% Compute autocovariance 
xcsamp = xcorr(mt-mean(mt),nlags, 'unbiased');

%% Make plots 

% Plot autocovariance
subplot(211);
h = plot(-nlags:nlags,kfunplot,-nlags:nlags,xcsamp, '--');
set(h(2), 'linewidth', 2);
legend('true', 'sample');
title('true and sample covariance');

% Plot  sample
subplot(212); nplot = min(1000,T); 
plot(1:nplot,mt(1:nplot)); 
title('sample m(t)');
xlabel('time (bins)'); ylabel('m(t)');


%% Compute ML estimate

prstrue = [rho,lm,pm,sigr]'; % true params
LB = [.01,.5, .1, .01]'; % lower bound
UB = [2*var(mt),2*length(mt),2,std(mt)]'; % upper bound

% Center the data so we don't need to fit the mean
mt_ctr = mt-mean(mt);

% make anonymous function (function pointer) for neg log-likelihood function.
lfun = @(prs)neglogli_GP(prs,kfun,mt_ctr);

% Set initial params
prs0 = prstrue + randn(4,1)*.25;  % randomize initial params a little bit
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB

% Compare true to initial params (useful for debugging)
[lfun(prstrue), lfun(prs0)] 

% set optimization options
opts = optimset('display','iter');

% optimize
prsML = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

% plot results
kfunplotML = kfun(xx,prsML(1),prsML(2),prsML(3),prsML(4));

subplot(111);
tt = -nlags:nlags;
h = plot(xx,kfunplot,xx,xcsamp,xx,kfunplotML, '--');
set(h(2:3), 'linewidth', 2);
legend('true', 'sample', 'ML fit');
title('true, sample, & ML-fit covariance');
xlabel('lag (sample)');
ylabel('autocovariance');


