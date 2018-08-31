def MotionCorrection_smooth_yy(R,G,a0,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D):    
    """MotionCoRection_smooth_yy - estimate activity y(t) under GP prior on m(t)
       Computes MAP estimate of yy under the following model:
        mm ~ N(mu_m, Sigma_m)  # motion artifact
        yy ~ N(mu_y, rho_y*I)  # neural activity-relatd fluorescence
        aa = D\yy             # aa comes from yy via AR1 process 
        R = mm + noise        # measured rfp
        G = aa.*mm + noise    # measured gcamp
        Sigma_m = Ubasis*(sdqrt^2)*Ubasis' -- low-rank approximation to covariance

     Inputs:
        R      [Tx1]  - rfp measurements
        G      [Tx1]  - gcamp measurements
        a0     [Tx1]  - initial value for a(t)
        rho_r  [1]   - variance of rfp noise
        rho_g  [1]   - variance of gfp noise
        mu_y   [1]   - prior mean of y(t)
        rho_y  [1]   - prior variance of y(t)
        mu_m   [1]   - mean of motion artifact
        Ubasis [Txk]  - basis for rank-k approx to variance of motion artifact
        Sinv   [kxk]  - Sparse diagonal matrix with truncated inverse singular values of Cm
        D      [TxT]  - mapping from neural activity yy to activity-related fluorescence aa

     Output: 
         yy [Tx1] - estimate of neural activity"""
    
    lfunc = lambda y : Loss_MotionCoRection(y,R,G,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D)
    aa = minimize(lfunc,a0)
    return(aa.x)

def bsxfunmult (a,b):
    temp = np.zeros_like(a)
    for i in range(len(b)):
        temp[i,:] = a[i,:]*b[i]
    return(temp)

# ============ Loss function =====================================
def Loss_MotionCoRection(yy,R,G,rho_r,rho_g,mu_y,rho_y,mu_m,Um,Sminv,D):
    aa = np.linalg.solve(D,yy)
    
    
    # Log-determinant term
    dvec = (1/rho_r + 1./rho_g*(aa**2))

    M = Sminv + Um.T.dot(bsxfunmult(Um, dvec))
    (_,Mlogdet) = np.linalg.slogdet(M)
    trm_logdet = .5*(Mlogdet)


    # Diag term
    trm_diag = .5*(sum((R-mu_m)**2)/rho_r + sum((G-mu_m*aa)**2)/rho_g)

    # Quad term
    xt = Um.T.dot((R-mu_m)/rho_r + aa*(G-mu_m*aa)/rho_g)
    trm_quad = -.5*(xt.T.dot(np.linalg.solve(M,xt)))

    # Prior term
    trm_prior = .5/rho_y*(sum((yy-mu_y)**2))

    # Sum them up
    obj = trm_logdet + trm_diag + trm_quad + trm_prior
    return (obj)