import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize 
from simulate_activity import cov, create_sigma

def sq_err(params, x, y):
    """ Helper function for  est_MSE that calculates the error"""
    rho, l, p= params
    output = cov(x, rho, l , p)
    error = np.sum([(y[i] - output[i])**2 for i in range(len(x)) if x[i] != 0])
    return error

def est_MSE_R(xx, R_corr, method='Powell'):
    """
    A least squared errors estimation of autocorrelation of R.

    Input:
        xx: indicies e.g: -2,-1,0,1,2
        R_corr: R(t) autocorrelation where R(t) has already been mean-centered
        method: Algorithm for minimization, default: Powell

    Returns:
        [rho, l, p, sig_r]

    """

    # Fit curve
    epsilon = 0.001
    bnds = ((epsilon,None),(epsilon,None),(epsilon,2-epsilon))
    mse_res = minimize(sq_err, [1,20,1], args = (xx,R_corr),method=method)

    # Print and generate new parameters

    inf_sig_r2 = R_corr[int(len(R_corr)/2)] - mse_res.x[0]
    inf_params = np.concatenate((mse_res.x,np.reshape(inf_sig_r2,(1,))))

    return inf_params

def neg_log_likelihood_short(param,r, toPrint = False):
    '''
    Function to minimize the negative log likelihood for short timescales.
    Computes the full covariance matrix.
    
    Must also provide r(t) in shape (len(r),1)
    '''
    # Read in parameters
    [rho,l,p,sig_r] = param
    sig_r2 = sig_r**2
    
    # Create intemediary variables
    lag = len(r)
    sig_m = create_sigma(rho,l,p,lag) + sig_r2*np.eye(lag)

    # Calculate log(det(Sigma_m + Sig_r2))
    (_,log_det_sig) = np.linalg.slogdet(sig_m)
    
    # calculate neg log-likelihood
    ret = .5*(log_det_sig + np.transpose(r).dot(np.linalg.solve(sig_m,r))[0,0])
    
    if toPrint:
        print("Rho: %1.3f, L: %1.3f, P: %1.3f, sig_r: %1.3f, %1.3f" %(rho, l, p,sig_r,ret))
    return ret

def est_MLE_R(R, initial, toPrint = False,method='SLSQP'):
    """
    A maximum likelihood approach to estimation of autocorrelation of R.

    Input:
        R: mean-centered R(t)
        initial: initial parameters for [rho, l, p, and sig_r] in the minimization
        toPrint: Boolean for printing

    Returns:
        [rho, l, p, sig_r]

    """
    epsilon = 0.01
    bnds = ((epsilon,None),(epsilon,None),(epsilon,2-epsilon),(0, None))

    # Reshape to feed to neg_log_likelihood_short
    R = np.reshape(R,(len(R),1))

    mle_res = minimize(neg_log_likelihood_short, args=(R,toPrint),x0=initial,bounds=bnds,method=method)
    return(mle_res.x)

def create_fitted(xx, params):
    """
    Generates the autocorrelation of R given parameters

    Input:
        xx: indicies e.g: -2,-1,0,1,2
        params: [rho, l, p, sig_r]
    
    Returns:
        fitted: autocorrelation of R
    """
    fitted = cov(xx,params[0],params[1],params[2])
    # put b in
    fitted[int(len(fitted)/2)] = fitted[int(len(fitted)/2)] + params[3]
    return(fitted)
