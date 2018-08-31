"""
Simulate activity (y and a where a is the timelagged version of y) and motion artifact, and thus RFP and GCaMP

example:
    Rho, L, P, T, Lag, Mu_m, Sig_r,alpha = 1,15,1.2,1000,10,10,.25,.9
    s = SimulateWorm(Rho = Rho,L = L, P = P, T = T, Lag = Lag, Mu_m= Mu_m, Sig_r=Sig_r,alpha = alpha)
    My_y,Sig_y2, L_y = 1,.1,2
    A = s.gen_A_t(My_y, Sig_y2, L_y)
"""
import numpy as np
from cmath import sqrt
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
import seaborn as sns

def rotate_left(l,n):
    return l[n:] + l[:n]

def triangle(lags, l):
    a = np.arange(l-lags, l)
    b = np.arange(l, l-lags-1, -1)
    c = np.hstack((a,b))
    return c

# Helper functions
def create_sigma (rho, l, p, length,debug = False):
    x = np.arange(0, length, 1)
    autocovar = cov(x,rho,l,p)
    toe = toeplitz(autocovar)
    if debug:
        f,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(autocovar)
        ax[0].set_title("Autocovaraince M_t")
        ax[1].imshow(toe)
        ax[1].set_title("Autocovaraince Toeplitz M_t")

    return toe

def cov(xx,r,l,p):
    """ Calculates covariance. Used for Fourier implementation"""
    to_ret = [(r*np.exp(-np.abs(x/l)**p)) for x in xx]
    return(to_ret)

def calc_corr(a,b,nlags):
    """Finds covariance between a and b assuming they have the same length.
    
    Takes lags up to nlags.
    
    Returns xx for plotting purposes and corr for correlation purposes
    """
    a = a-np.mean(a)
    b = b-np.mean(b)
    T = len(a)
    #nlags = int(np.ceil(L*6*.1)*10);# number of lags for plotting purposes
    xx = list(range(-nlags,nlags+1))
    
    xcsamp = np.correlate(a,b,'full')[T-1-nlags:T+nlags]
    corr = np.divide(xcsamp,triangle(nlags,T))
    return (xx, corr)

class SimulateWorm:
    """generating the full matrix covariance. Not recommend to use more than ___ data points."""

    def __init__(self, Rho, L, P, T, Lag, Mu_m, Sig_r, alpha):
        self.Rho = Rho
        # length-scale
        self.L = L
        self.P = P
        self.T = T
        self.Lag = Lag
        self.alpha = alpha
        
        # m(t) ∼ N (mu_m, sig_m) 
        self.M_t = None
        self.Mu_m = Mu_m
        self.Sigma_m = None
        
        # R_t = M_t + e where e ~ N(0,sig_r^2)
        self.R_t = None
        self.R_m = Mu_m
        self.Sig_r = Sig_r
        
        # Activity
        self.Y_t = None
        self.A_t = None

        # helper functions
        self.nlags = int(np.ceil(L*6*.1)*10);# number of lags for plotting purposes
        self.xx = list(range(-self.nlags,self.nlags+1))
        self.tvec = None
        self.xcsamp = None


    def gen_M_t(self,method='fourier',debug=False):
        """Generate motion artifact from $\Sigma_m = \rho e^{-|\tau/a|^p}$"""
        if method == 'matrix':
            if self.T > 1000:
                raise Exception('Time too long for Matrix implementation! Please use Fourier')
            mu = np.repeat(self.Mu_m,self.T)
            self.Sigma_m = create_sigma(self.Rho, self.L, self.P, self.T,debug = debug)
            self.M_t = np.random.multivariate_normal(mu, self.Sigma_m) # motion rv
        else:
            kfunplot = cov(self.xx,self.Rho,self.L,self.P)
            # Evaluate covariance function and take FFT
            self.tvec = np.concatenate((range(int(np.floor((self.T-1)/2))+1),range(int(-np.ceil((self.T-1)/2)),0))) # vector of time bins of length T
            kf = cov(self.tvec,self.Rho,self.L,self.P) # covariance function of appropriate length
            kfh = [sqrt(x) for x in (np.fft.fft(kf))] # take fourier transform

            # Generate sample
            self.M_t = np.real(np.fft.ifft(kfh*np.fft.fft(np.random.normal(0,1,self.T))))+self.Mu_m
        return(self.M_t)

    def gen_R_t(self,method='fourier',recal=True):
        if recal or self.M_t is None:
            self.gen_M_t(method)
            
        self.R_t = self.M_t + self.Sig_r*np.random.normal(0,1,size=self.T)
        return (self.R_t)
    
    def gen_Y_t(self, mu_y, sig_y2, l_y):
        """ Generates neural activity, y, from GP"""
        tau_g = -1/np.log(self.alpha) # time constant of gcamp decay (in time bins)
        
        Mu_y = np.repeat(mu_y,self.T) # mean of neural activity y
        cov_kernel = cov(list(range(1,self.T+1)),sig_y2, l_y, 1) # true covaraiance kernel
        sqrtSy = [sqrt(x) for x in (np.fft.fft(cov_kernel))]
        
        self.Y_t = np.real(np.fft.ifft(sqrtSy*np.fft.fft(np.random.normal(0,1,self.T))))+Mu_y
        return (self.Y_t)
    
    def gen_A_t(self,mu_y,sig_y2,l_y, recal=True):
        if recal or self.Y_t is None:
            self.gen_Y_t(mu_y, sig_y2, l_y)
        
        # Build D
        
        row_ind = np.concatenate((np.array(range(self.T)),np.array(range(1,self.T))))
        col_ind = np.concatenate((np.array(range(self.T)),np.array(range(self.T-1))))
        data = np.concatenate((np.repeat(1,self.T),np.repeat(-self.alpha,self.T-1)))
        D = csr_matrix((data, (row_ind, col_ind)),shape=(self.T,self.T)).toarray()

        # Find A
        self.A_t = spsolve(D,self.Y_t)
        return(self.A_t)

    def gen_G_t(self, mu_y,sig_y2,l_y,sig_g,recal = True):
        if recal or self.A_t is None:
            self.gen_A_t(mu_y,sig_y2,l_y)
        if recal or self.M_t is None:
            self.gen_M_t('fourier')
        self.Sig_g = sig_g
        
        
        self.G_t = (self.M_t*self.A_t) + self.Sig_g*np.random.normal(0,1,size=self.T)
        return(self.G_t)
