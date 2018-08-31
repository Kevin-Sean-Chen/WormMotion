function aa = MotionCorrection_smooth_aa(rr,gg,a0,rho_r,rho_g,mu_a,rho_a,mu_m,Um,Sminv,D)
% MotionCorrection_smooth_aa - estimate fluorescence a(t) under GP prior on m(t)
%
% Computes MAP estimate of yy under the following model:
%    mm ~ N(mu_m, Sigma_m)  % motion artifact
%    aa ~ N(mu_a, rho_a*I)  % neural activity-relatd fluorescence
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
%    mu_a   [1]   - prior mean of a(t)
%    rho_a  [1]   - prior variance of a(t)
%    mu_m   [1]   - mean of motion artifact
%    Ubasis [Txk]  - basis for rank-k approx to variance of motion artifact
%    Sinv   [kxk]  - Sparse diagonal matrix with truncated inverse singular values of Cm
%    D      [TxT]  - mapping from neural activity yy to activity-related fluorescence aa
%
% Output: 
%     yy [Tx1] - estimate of neural activity

lfunc = @(y) Loss_MotionCorrection(y,rr,gg,rho_r,rho_g,mu_a,rho_a,mu_m,Um,Sminv);
opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
aa = fminunc(lfunc,a0,opts);
end

% ============ Loss function =====================================
function obj = Loss_MotionCorrection(aa,rr,gg,rho_r,rho_g,mu_a,rho_a,mu_m,Um,Sminv)

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
trm_prior = .5/rho_a*sum((aa-mu_a).^2);

% Sum them up
obj = trm_logdet + trm_diag + trm_quad + trm_prior;
    
end