
%% Set conditions of recordings
n = 5; 
T = 80;
t = 0:T-1;
mlength = 4;
mrho = .5;
alength = 3;
arho = .1;
mu0 = 5;
mu_m = exprnd(mu0,[n 1]);
mu_a = exprnd(mu0,[n 1]);
% b_g = exprnd(mu0/4,[n 1]);
b_g = 2*ones(n,1);

rnsevar = 4;
gnsevar = 4;

%% Simulate latent states
mCt = toeplitz(mrho*exp(-(t/mlength).^2/2));
Lm = sprandn(n,n,.1) + eye(n);
mCn = inv(Lm*Lm');
aCt = toeplitz(arho*exp(-(t/alength).^2/2));
La = sprandn(n,n,.05) + eye(n);
aCn = inv(La*La');

m = mvnrnd(repmat(mu_m,T,1),kron(mCt,mCn));
M = reshape(m,n,T);
figure;imagesc(M);title('Motion')
a = mvnrnd(repmat(mu_a,T,1),kron(aCt,aCn));
A = reshape(a,n,T);
figure;imagesc(A);title('Activity')
figure;imagesc(A.*M);title('product')
figure;
subplot(211);plot(t,M(1,:),t,A(1,:))
legend('motion','activity')
subplot(212);plot(t,M(1,:).*A(1,:))


%% Simulate observations
R = M + randn(n,T)*sqrt(rnsevar);
G = A.*M + repmat(b_g,1,T) + randn(n,T)*sqrt(rnsevar);
figure;plot(t,R(1,:),'r',t,G(1,:),'g')
% legend('RFP','GcAMP')

figure;imagesc(R);title('RFP')
figure;imagesc(G);title('GcAMP')