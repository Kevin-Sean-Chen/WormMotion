% Simple script to try out fitting parameters on real worm data

% Load raw data for neurons 9 and 12  (Change path for diff users)
load /Users/pillow/Documents/matlab/rawdata/CelegansData/neurons9_12.mat

% Set which neuron to use
cell9 = 1;
if cell9;
    mt = n9';
else
    mt = n12';
end
std0 = std(mt);
mt = mt./std0;  % standardize to have unit standard deviation

% Compute autocovariance 
nlags = 300;
xx = -nlags:nlags;
xcsamp = xcorr(mt-mean(mt),nlags, 'unbiased');
plot(xx,xcsamp, '.-');
xlabel('lag'); ylabel('cross-cov');
title('raw cross-cov');


%% Hand-tune the GP params (bye eye)

% Define true GP covariance function
kfun = @(x,r,l,p,sigr)(r*exp(-abs(x/l).^p)+(x==0).*sigr^2); % anonymous function for cov

% Set initial values for hyperparams
if cell9
        rho = .85;  % prior variance of m
        lm = 13; % length scale
        pm = .6; % power  (note: only valid covariance function for p<=2)
        sigr = .4;
    
else        
        rho = .9;  % prior variance of m
        lm = 45; % length scale
        pm = .9; % power  (note: only valid covariance function for p<=2)
        sigr = .3;
end

kfunplot = kfun(xx,rho,lm,pm,sigr);
clf; plot(xx,xcsamp,'-o', xx,kfunplot,'-x');
title('autocovariance'); xlabel('lag');
legend('sample', 'GP');


%% Compute ML estimate

prsSet0 = [rho,lm,pm,sigr]'; % true params
LB = [.01,.5, .1, .01]'; % lower bound
UB = [2*var(mt),2*length(mt),2,std(mt)]'; % upper bound

% Center the data so we don't need to fit the mean
mt_ctr = mt-mean(mt);

% make anonymous function (function pointer) for neg log-likelihood function.
lfun = @(prs)neglogli_GP(prs,kfun,mt_ctr);

% Set initial params
prs0 = prsSet0 + randn(4,1)*.1;  % randomize initial params a little bit
prs0 = max([prs0';LB'])'; % make sure it didn't go below LB or UB
prs0 = min([prs0';UB'])'; % make sure it didn't go below LB or UB

% Compare true to initial params (useful for debugging)
[lfun(prsSet0), lfun(prs0)] 

% set optimization options
opts = optimset('display','iter');

% optimize
prsML = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

%% Plot results
kfunplotML = kfun(xx,prsML(1),prsML(2),prsML(3),prsML(4));

subplot(211);  % many lags
tt = -nlags:nlags;
h = plot(xx,xcsamp,xx,kfunplotML, 'o-');
set(h(1), 'linewidth', 3);
set(h(2), 'linewidth', 1,'markersize', 4);
legend('sample', 'ML fit');
title('sample & ML-fit covariance');
xlabel('lag');
ylabel('autocovariance');

subplot(212)  % Zoomed in 
tt = -nlags:nlags;
h = plot(xx,xcsamp,xx,kfunplotML, 'o-');
set(h(1), 'linewidth', 3);
set(h(2), 'linewidth', 1,'markersize', 4);
legend('sample', 'ML fit');
title('(zoomed in)');
xlabel('lag');
ylabel('autocovariance');
set(gca,'xlim', [-50 50]);
