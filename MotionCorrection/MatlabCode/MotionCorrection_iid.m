function aa = MotionCorrection_iid(rr,gg,rho_r,rho_g,mu_m,rho_m,mu_a,rho_a)
% MOTIONCORRECTION_IID - motion correction estimate under iid priors on m and a
%
% Computes MAP estimate of aa under the following model:
%    mm ~ N(mu_m, rho_m*I)  % motion artifact
%    aa ~ N(mu_a, rho_a*I)  % activity-related fluorescence
%    rr = mm + noise        % measured rfp
%    gg = aa.*mm + noise    % measured gcamp
%
% Inputs:
%    rr [Tx1] - rfp measurements
%    gg [Tx1] - gcamp measurements
%  rho_r [1]  - variance of rfp noise
%  rho_g [1]  - variance of gfp noise 
%   mu_m [1]  - mean of motion artifact
%  rho_m [1]  - variance of motion artifact
%   mu_a [1]  - mean of activity-fluorescence
%  rho_a [1]  - variance of activity-fluorescence
%
% Output: 
%     aa [Tx1] - estimate of neural activity-related fluorescence

aa0 = gg./rr; % initial estimate of activity
T = length(gg); % number of timesteps
Crg = diag([rho_r, rho_g]); % covariance for measurement noise

opts = optimoptions('fmincon', 'display', 'notify');
lb = 1e-2; % lower bound on a(t)
ub = inf;  % upper bound on a(t)

% Optimize for each time bin 
% (could parallelize with parfor, since independent for each bin!)
aa = zeros(T,1);
fprintf('MotionCorrection_iid: estimating time bin\n ');
for jj = 1:T
    if mod(jj,10)==0
        fprintf('%d ',jj);
    end
  % set loss function
    lfunc = @(a) neglogli_MotionCorrectionIID(a,[rr(jj);gg(jj)],Crg,mu_m,rho_m,mu_a,rho_a);
    % optimize for this time bin (constraining a(t)>0)
    aa(jj) = fmincon(lfunc,aa0(jj),[],[],[],[],lb,ub,[],opts);
end
fprintf('\n');
end

%% ---- Loss function ---------
function negL = neglogli_MotionCorrectionIID(a,rg,Crg,mu_m,rho_m,mu_a,rho_a)

% Unpack a for likelihood terms
va = [1;a]; 
Ca = Crg + rho_m*(va*va');
rgctr = rg-va*mu_m;

% likelihood terms
trm_logdet = .5*logdet(Ca);  % note logdet should be in your path!
trm_quad = .5*rgctr'*(Ca\rgctr);

% prior terms
trm_logprior = .5/rho_a*(a-mu_a)^2;

% Combine terms
negL = trm_logdet+trm_quad+trm_logprior;

end

    
