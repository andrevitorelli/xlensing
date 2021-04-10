import numpy as np

def ln_gaussian_likelihood_maker(data,model,hartlap=1):
    
    def ln_gaussian_like(theta):
        signal, covariance = data
        N = len(signal)

        model_eval = model(theta)#model is  m =  f(theta,x) with x previously set by functools.partial


        inv_cov_estim = hartlap*np.linalg.inv(covariance)
        vec =  signal - model_eval

        vec_row = vec.reshape(1,N)
        vec_col = vec.reshape(N,1)

        lndetcov = np.log(np.linalg.det(covariance))#data['Delta_sigma_cov_stick12']/1e30)) +(len(radii)+4)*np.log(1e30)

        result = -(np.dot(vec_row,np.dot(inv_cov_estim,vec_col)))-N*1.837877-abs(lndetcov)#1.837877 = ln(2pi)

        if result != result: #tests if result is nan
            result = -np.inf
        return result/2
    return ln_gaussian_like


def ln_flat_prior_maker(low,high,pos):
    def prior(theta):
        if low < theta[pos] < high:
            return 0
        else:
            return -np.inf
    return prior

def ln_gaussian_prior_maker(mu, sigma, pos):
    def prior(x):
        result = -(x[pos] - mu)*(x[pos] - mu)/(sigma**2)\
                 -1.837877\
                 -np.log(sigma)
        return result/2
    return prior