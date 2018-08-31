function y = MotionCorrection(rr,gg,y0,rho_r,rho_g,mu_y,rho_y,mu_m,Ubasis,sdiag,D)
% MOTIONCORRECTION - motion correction estimate under GP priors on m
%
% Computes MAP estimate of yy under the following model:
%    mm ~ N(mu_m, Sigma_m)  % motion artifact
%    yy ~ N(mu_y, rho_y*I)  % neural activity
%    aa = D\yy
%    rr = mm + noise        % measured rfp
%    gg = aa.*mm + noise    % measured gcamp
%    Sigma_m = Ubasis*sdiag*Ubasis' -- low-rank approximation to covariance
%
% Inputs:
%    rr     [Tx1]  - rfp measurements
%    gg     [Tx1]  - gcamp measurements
%    y0     [Tx1]  - initial value for neural activity estimate
%    rho_r  [1]   - variance of rfp noise
%    rho_g  [1]   - variance of gfp noise
%    mu_y   [1]   - prior mean of neural activity
%    rho_y  [1]   - prior variance of neural activity 
%    mu_m   [1]   - mean of motion artifact
%    Ubasis [Txk]  - basis for rank-k approx to variance of motion artifact
%    sdiag  [kxk]  - Diagonal matrix with truncated singular values of Sigma_m
%    D      [TxT]  - mapping from neural activity yy to activity-related fluorescence aa
%
% Output: 
%     yy [Tx1] - estimate of neural activity

lfunc = @(y) Loss_MotionCorrection(y,rr,gg,rho_r,rho_g,mu_y,rho_y,mu_m,Ubasis,sdiag,D);
opts = optimoptions('fminunc', 'display', 'iter', 'algorithm', 'quasi-newton');
y = fminunc(lfunc,y0,opts);
end

% ============ Loss function =====================================
function obj = Loss_MotionCorrection(y,rr,gg,rho_r,rho_g,mu_y,rho_y,mu_m,Ubasis,sdiag,D)

Tb = size(sdiag,1);

trm_prior =  1/(2*rho_y)*((y-mu_y)'*(y-mu_y)); % prior term

trm_logdet = 0.5*logdet(1/rho_r*sdiag + 1/rho_g*sqrt(sdiag)*Ubasis'*diag(D\y)*diag(D\y)*Ubasis*sqrt(sdiag) + eye(Tb)); % logdet term in posterior on yy

trm_quad= -0.5*(sqrt(sdiag)*Ubasis'*(rr./rho_r + diag(D\y)*gg./rho_g) + sqrt(sdiag)\Ubasis'*mu_m)'*...
    ((1/rho_r*sdiag + 1/rho_g*sqrt(sdiag)*Ubasis'*diag(D\y)*diag(D\y)*Ubasis*sqrt(sdiag) + eye(Tb))\eye(Tb))*...
    (sqrt(sdiag)*Ubasis'*(rr./rho_r + diag(D\y)*gg./rho_g) + sqrt(sdiag)\Ubasis'*mu_m); % likelihood term

obj = trm_prior + trm_quad + trm_logdet;
    
end