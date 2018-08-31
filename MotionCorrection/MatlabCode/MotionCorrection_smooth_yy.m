function aa = MotionCorrection_smooth_yy(rr,gg,a0,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D)
% MotionCorrection_smooth_yy - estimate activity y(t) under GP prior on m(t)
%
% Computes MAP estimate of yy under the following model:
%    mm ~ N(mu_m, Sigma_m)  % motion artifact
%    yy ~ N(mu_y, rho_y*I)  % neural activity-relatd fluorescence
%    aa = D\yy;             % aa comes from yy via AR1 process 
%    rr = mm + noise        % measured rfp
%    gg = aa.*mm + noise    % measured gcamp
%    Sigma_m = Ubasis*(sdqrt^2)*Ubasis' -- low-rank approximation to covariance
%
% Inputs:
%    rr     [Tx1]  - rfp measurements
%    gg     [Tx1]  - gcamp measurements
%    a0     [Tx1]  - initial value for a(t)
%    rho_r  [1]   - variance of rfp noise
%    rho_g  [1]   - variance of gfp noise
%    mu_y   [1]   - prior mean of y(t)
%    rho_y  [1]   - prior variance of y(t)
%    mu_m   [1]   - mean of motion artifact
%    Ubasis [Txk]  - basis for rank-k approx to variance of motion artifact
%    Sinv   [kxk]  - Sparse diagonal matrix with truncated inverse singular values of Cm
%    D      [TxT]  - mapping from neural activity yy to activity-related fluorescence aa
%
% Output: 
%     yy [Tx1] - estimate of neural activity

lfunc = @(y) Loss_MotionCorrection(y,rr,gg,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D);
opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
fprintf('initial negative log-likelihood: %1.5f \n',lfunc(a0))
aa = fminunc(lfunc,a0,opts);
fprintf('final negative log-likelihood: %1.5f \n',lfunc(aa))
end

% ============ Loss function =====================================
function obj = Loss_MotionCorrection(yy,rr,gg,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D)

aa = D\yy;

% Log-determinant term
dvec = (1/rho_r + 1./rho_g*aa.^2);
M = Sminv + Um'*bsxfun(@times, Um, dvec);
trm_logdet = .5*logdet(M);

% Diag term
trm_diag = .5*(sum((rr-mu_m).^2)/rho_r + sum((gg-mu_m.*aa).^2)./rho_g);

% Quad term
xt = Um'*((rr-mu_m)/rho_r + aa.*(gg-mu_m.*aa)/rho_g);
trm_quad = -.5*xt'*(M\xt);

% Prior term
trm_prior = .5/rho_y*sum((yy-mu_y).^2);

% Sum them up
obj = trm_logdet + trm_diag + trm_quad + trm_prior;
    
end