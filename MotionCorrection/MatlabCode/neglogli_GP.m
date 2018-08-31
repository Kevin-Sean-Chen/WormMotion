function negL = neglogli_GP(prs,kfun,x)
% negL = neglogli_GP(prs,kfun,x)
%
% Compute maximum marginal likelihood estimate for GP hyperparameters given
% a sample (assumed to be on a grid from [1 length(mt)]
%
% Inputs:
%        prs = params of kernel function
%       kfun = function handle pointing to kernel (covariance) function
%  x [T x 1] = the zero-mean sample.
%
% Output:
%       negL = negative log-likelihood

tvec = (0:(length(x)-1))';  % vector of time bins 
kvec = kfun(tvec,prs(1),prs(2),prs(3),prs(4)); % 1st row of covariance matrix
C = toeplitz(kvec); % covariance matrix

% Compute likelihood
negL = .5*logdet(C) + .5*(x'*(C\x));


