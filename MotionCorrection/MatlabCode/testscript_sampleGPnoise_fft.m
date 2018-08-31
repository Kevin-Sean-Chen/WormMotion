% Simple script to illustrate efficient sampling from GP using the
% convolution theorem (i.e., in the fourier domain).

% Define true GP covariance function
rho = 5;  % prior variance of m
lm = 8; % length scale
pm = .75; % power  (note: only valid covariance function for p<=2)
kfun = @(x,r,l,p)(r*exp(-abs(x/l).^p)); % anonymous function for cov

% Define mean
mu = 10;

% Plot it 
%nlags = ceil(lm*4*.1)*10;  % number of lags for plotting purposes
nlags = 50;
xx = -nlags:nlags;
kfunplot = kfun(xx,rho,lm,pm);
subplot(211);
plot(xx,kfunplot);
title('true covariance'); xlabel('lag');

%% Make correlated noise with this covariance function
T = 1e5;  % make 100K samples

% Evaluate covariance function and take FFT
tvec = ([0:floor((T-1)/2),-ceil((T-1)/2):-1])';  % vector of time bins of length T
kf = (kfun(tvec,rho,lm,pm)); % covariance function of appropriate length
kfh = real(fft(kf)); % take fourier transform

% Generate sample
mt = real(ifft(sqrt(kfh).*fft(randn(T,1))))+mu; 

% Let's check its marginal statistics:
fprintf('mu: %.2f   sample mu: %.2f\n', mu, mean(mt));
fprintf('var: %.2f   sample var: %.2f\n', rho, var(mt));

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
subplot(212); nplot = 1000; 
plot(1:nplot,mt(1:nplot)); 
title('sample m(t)');
xlabel('time (bins)'); ylabel('m(t)');

