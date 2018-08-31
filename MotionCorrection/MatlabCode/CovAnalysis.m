% Short script to test analytical results regarding correlations between
% RFP and GcAMP that follow from our model assumptions
clear all;close all
%% Set recording parameters
T = 500; %length of recording
Ta = 10;%lag between activity and motion
t = 0:T-1;
%% Set parameters for coupled GP's
% Set means of tiem series
mu_a = 3;% Mean activity
mu_m = 2;% mean motion artifact

% Set parameters for convolution kernels of coupled components
mlenc = 10;%motion coupled length scale
mrhoc = sqrt(.3);%motion coupled standard deviation
alenc = 10;%activity autocovariance length scale
arhoc =sqrt(.3);%activity marginal standard deviation
lagm = 10;

% Set parameters for convolutions kernels of independent components
mrhoi = 0;%sqrt(1-mrhoc^2);
mleni = .8*mlenc;
arhoi =0;%sqrt(1-arhoc^2);
aleni = .5*alenc;

% Define convolution kernels
km = mrhoc*exp(-(t-lagm).^2/mlenc^2/2);
ka = arhoc*exp(-t.^2/alenc^2/2);
hm = mrhoi*exp(-(t).^2/mleni^2/2);
ha = arhoi*exp(-(t).^2/aleni^2/2);

figure;
subplot(221);plot(km);title('k_m')
subplot(222);plot(hm);title('h_m')
subplot(223);plot(ha);title('h_a')
subplot(224);plot(ka);title('k_a')

% Define functions that make covariances
cov_ii = @(rho,len,tt)(sqrt(pi)^-1*rho^2*exp(-tt.^2/len^2/4)/len);
covamfun = @(arho,mrho,alen,mlen,Sig,lag,tt)(sqrt(2*pi)^-1*arho*mrho*exp(-(tt-lag).^2/Sig/4)/sqrt(alen^2+mlen^2)); % anonymous function for cov

% Set parameters of covariance kernels based on convolution kernel
% parameters
Sigma_am = alenc^2*mlenc^2/(alenc^2+mlenc^2);
cov_mc = cov_ii(mrhoc,mlenc,t);
cov_ac = cov_ii(arhoc,alenc,t);
cov_mi = cov_ii(mrhoi,mleni,t);
cov_ai = cov_ii(arhoi,aleni,t);
cov_ma = covamfun(arhoc,mrhoc,alenc,mlenc,Sigma_am,lagm,t);
cov_am = covamfun(arhoc,mrhoc,alenc,mlenc,Sigma_am,lagm,-t);

% Define auto- and cross-covariance sequences
Covm = cov_mc + cov_mi;
Cova = cov_ac + cov_ai;
Covma = cov_ma;
Covam = cov_am;

figure;
subplot(221);plot(Covm);title('Cov_m')
subplot(222);plot(Cova);title('Cov_a')
subplot(223);plot(Covma);title('Cov_{ma}')
subplot(224);plot(Covam);title('Cov_{am}')

% % Make joint covariance matrices
Cm = toeplitz(Covm);
Ca = toeplitz(Cova);

Cma = zeros(T,T);Cam = Cma;
for tt = 1:T
    Cma(tt,:) = covamfun(arhoc,mrhoc,alenc,mlenc,Sigma_am,lagm,-tt+1+t);
    Cam(tt,:) = covamfun(arhoc,mrhoc,alenc,mlenc,Sigma_am,lagm,tt-1-t);
end

C = [Cm Cma;Cam Ca];
mu = [ones(T,1)*mu_m;ones(T,1)*mu_a];
figure;imagesc(C)
%% Generate latent states

% Condition covariance
[U,lamb] = eig(C);
l = diag(lamb);
l(l<1e-10*max(l))=0;
CC = U*diag(l)*U';
figure;imagesc(CC)

x = mvnrnd(zeros(2*T,1),CC);
m = x(1:T);
a = x(T+1:end);

figure;
subplot(211);plot(t,m,t,a)
legend('motion','activity')

empRma = crosscorr(a,m);
subplot(212);plot(empRma)
%% Generate observations

% Set noise variance for observations
rnsestd = mu_m/2;
gnsestd = mu_m/3;
r = m + randn(1,T)*rnsestd;
g = a.*m + randn(1,T)*gnsestd;
figure;
subplot(211);plot(t,r)
subplot(212);plot(t,g)
% plot(t,r,'r',t,g,'g')
% legend('RFP','GcAMP')
return
%% Empirical covariances
empRr = autocorr(r);
figure;plot(1:length(empRr),empRr,1:length(empRr),Rm(1:length(empRr))/(rnsevar+mrho)+rnsevar/(rnsevar+mrho)*[1 zeros(1,length(empRr)-1)])

figure;plot(crosscorr(G,R));title('R G crosscorr')
figure;plot(autocorr(R));title('R autocorr')
figure;plot(autocorr(G));title('G autocorr')

%% Theoretical covariances
delta = zeros(length(Rm),1);%delta function
delta(round((length(Rm)-1)/2 + 1)) = 1;

RrT = Rm + delta'*rnsevar;
RgT = Ra*Rm + Ram.*flip(Ram) + mu_a*mu_a*Rm + mu_m*mu_m*Ra;


