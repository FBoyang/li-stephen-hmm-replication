from ntpath import join
from optparse import Values
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
from numba import jit
import scipy
# from scipy.special import logsumexp
from numpy import logaddexp
from scipy.stats import pearsonr
from tqdm import tqdm



# def trans_X(x,y,d,rho,k):
#     '''
#     Compute the transition probability P(X_{j+1}=y| X_j=x)
#     Args:
#         rho: recombination parameter 
#         d: physical distance between j and j+1 
#     Outputs:
#         probability of P(X_{j+1}=x'| X_j=x)
#     '''
#     if x == y:
#         return np.exp(-rho*d/k)+(1-np.exp(-rho*d/k))(1/k)
#     else:
#         return (1-np.exp(-rho*d/k))*(1/k)



# @jit(nopython=True)
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
        .. versionadded:: 0.16.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2
    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    """
    # a = _asarray_validated(a, check_finite=False)
    # a = a.flatten()

    b = np.array(b)
    # a = a*b
    # if b is not None:
    #     a, b = np.broadcast_arrays(a, b)
    #     if np.any(b == 0):
    #         a = a + 0.  # promote to at least float
    #         a[b == 0] = -np.inf
    a = np.array(a)
    a_max = np.max(a)
    # if a_max.ndim > 0:
    # a_max[~np.isfinite(a_max)] = 0
    if not np.isfinite(a_max):
        a_max = 0

    # if b is not None:
    b = np.asarray(b)
    tmp = b * np.exp(a - a_max)
    # else:
    #     tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    # with np.errstate(divide='ignore'):
    s = np.sum(tmp)
    # if return_sign:
    #     sgn = np.sign(s)
    #     s *= sgn  # /= makes more sense but we need zero -> zero
    out = np.log(s)

    # if not keepdims:
    #     a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    # if return_sign:
    #     return out, sgn
    # else:
    return out


@jit(nopython=True)
def gammax(theta, hs,h_origs,k):
    '''
    Compute the P(h_{k+1, j+1}| X_{j+1}=x,h1,..,h_k)
    Args;
        h: value of h_{k+1, j}
        h_orig: value of h given it comes from jth reference
        theta: error rate need to be estimated
    '''
    
    boolean_array = hs==h_origs
    intercept = np.ones(k)
    
    return intercept*0.5*theta/(k+theta)+boolean_array*k/(k+theta)

# @jit(nopython=True)
def gammax_log(logtheta, hs,h_origs,k):
    '''
    Compute the P(h_{k+1, j+1}| X_{j+1}=x,h1,..,h_k)
    Args;
        h: value of h_{k+1, j}
        h_orig: value of h given it comes from jth reference
        theta: error rate need to be estimated
    return: log gamma 
    '''
    
    boolean_array = (hs==h_origs) + 1e-10
    intercept = np.ones(k)
    return np.log(intercept) + np.log(0.5) + logtheta - 2*logsumexp(np.array([np.log(k),logtheta]),b=[1]*2)+np.log(boolean_array) + np.log(k) 

# def fill(theta, alphas0, j, M, k):
#     '''
#     j: Initial value should be 0
#     recursively solving alpha_{j+1}
#     '''
#     alphas = np.zeros(k)
#     # for i in range(k):
#     hs = H[ref_order[:k],j] # H is a k by S matrix with reference haplotypes
#     h_origs = H[ref_order[:k],j] # Assumption, the joint probability is added sequentially
#     gamma = gammax(theta, hs,h_origs,k)
#     # print(gamma)
#     d = Distlist[j] - Distlist[j-1] # Distlist records the physical distance at each loci
#     pj = np.exp(-rhos[j-1]*d/k)
#     if j == M-1: ## j means j+1 in the algorithm A5
#         alphas = gamma*(pj*alphas0 + (1-pj)*np.sum(alphas0)/k)
#         return alphas
#     else:
#         alphas = gamma*(pj*alphas0 + (1-pj)*np.sum(alphas0)/k)
#         alphas1 = fill(theta,alphas,j+1, M, k)
#         return alphas1



# @jit(nopython=True)
def fill_norec_long_nb(logtheta, alphas0, M, k,ref_order,testhap=np.array([])):
    '''
    j: Initial value should be 0
    recursively solving alpha_{j+1}
    Add more precision
    factor: a log based constant
    '''
    logalphas0 = np.log(alphas0)
    logalphas = np.zeros(k)
    for j in range(1, M):
        if testhap.size >0:
            hs = testhap[j]
        else:
            hs = H[ref_order[k],j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[ref_order[:k],j] 
        gamma = gammax(np.exp(logtheta),hs,h_origs,k) # times a constant factor to avoid underflow
        # print('loggamma is: ')
        # print(loggamma)
        # pj = np.exp(-(GM[j+1]-GM[j])/k)
        # log sum of exp(-(GM[j+1]-GM[j])/k + logalphas0) + sum (np.exp(logalphas0))
        # term0: -gm/k + logalphas0; 1 x k
        term0=(-(GM[j]-GM[j-1])/k+logalphas0).reshape(-1,1)
        # term1: (logalpha_i(x') - logk); 1 x k => need to be broadcast
        term1=(logalphas0-np.log(k)) # k terms, plus sign
        # add12 = np.logaddexp(term0, term1)
        # term2: logalpha_i(x') - logk - gm/k; 1 x k => need to be broadcast to k by k
        # term2=(logalphas0-np.log(k)-(GM[j]-GM[j-1]+1e-15)/k) # k terms, minus sign
        # term12 = scipy.special.logsumexp(np.concatenate((term1, term2)))
        sumlogalphas0 = logsumexp(logalphas0,b=[1]*len(logalphas0))
        sumlogalphas0 = sumlogalphas0 + logsumexp([0,(-GM[j]+GM[j-1]-1e-20)/k],b=[1,-1]) - np.log(k)
        # sumlogalphas0 = sumlogalphas0+np.log(1 - np.exp(-(GM[j]-GM[j-1]+1e-10)/k))-np.log(k)
        # print(sumlogalphas0)
        if np.isnan(sumlogalphas0):
            break
        logsumterms = np.concatenate((term0.reshape(-1,1), np.repeat(sumlogalphas0.reshape(1,-1),k,axis=0)),axis=1)
        # print(f'logsumterms shape is {logsumterms.shape}')
        logsumtermsnew = np.zeros(k)
        for i in range(k):
            logsumtermsnew[i] = logsumexp(logsumterms[i],b=[1]*len(logsumterms[i]))
        # logsumterms = logsumexp(logsumterms,axis=1).flatten()
        logalphas = logsumtermsnew + np.log(gamma)
        logalphas0 = logalphas

        # for w in range(k):
        #     logsumterms = np.concatenate((term0[w,:],term1,term2))
        #     # logsumterms = np.concatenate((term0.reshape(-1,1), np.repeat(term1.reshape(1,-1),k,axis=0), np.repeat(term2.reshape(1,-1),k,axis=0)))
        #     logalphas[w]=logsumexp(logsumterms,b=[1]*(k+1)+[-1]*k)+np.log(gamma[w])
        logalphas0 = logalphas
    return logalphas0



# @jit(nopython=True)
def fill_norec_long(logtheta, alphas0, M, k,ref_order,testhap=np.array([])):
    '''
    j: Initial value should be 0
    recursively solving alpha_{j+1}
    Add more precision
    factor: a log based constant
    '''
    logalphas0 = np.log(alphas0)
    for j in range(1, M):
        if testhap.size >0:
            hs = testhap[j]
        else:
            hs = H[ref_order[k],j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[ref_order[:k],j] 
        gamma = gammax(np.exp(logtheta),hs,h_origs,k) # times a constant factor to avoid underflow
        # print('gamma is: ')
        # print(gamma)
        # pj = np.exp(-(GM[j+1]-GM[j])/k)
        # log sum of exp(-(GM[j+1]-GM[j])/k + logalphas0) + sum (np.exp(logalphas0))
        # term0: -gm/k + logalphas0; 1 x k
        term0=(-(GM[j]-GM[j-1]+Genetic_dist)/k+logalphas0).reshape(-1,1) # -rho *d /k + logalpha_j-1
        # term1: (logalpha_i(x') - logk); 1 x k => need to be broadcast
        # term1=(logalphas0-np.log(k)) # k terms, plus sign
        # add12 = np.logaddexp(term0, term1)
        # term2: logalpha_i(x') - logk - gm/k; 1 x k => need to be broadcast to k by k
        # term2=(logalphas0-np.log(k)-(GM[j]-GM[j-1]+1e-15)/k) # k terms, minus sign
        # term12 = scipy.special.logsumexp(np.concatenate((term1, term2)))
        term1 = scipy.special.logsumexp(logalphas0 - np.log(k)) # log sum of alphas
        term2 = scipy.special.logsumexp(logalphas0-(GM[j]-GM[j-1]+Genetic_dist)/k - np.log(k))
        sumlogalphas0 = scipy.special.logsumexp([term1,term2],b=[1,-1])

        # sumlogalphas0 = sumlogalphas0+np.log(1 - np.exp(-(GM[j]-GM[j-1]+1e-10)/k))-np.log(k)
        # print(sumlogalphas0)
        # if np.isnan(sumlogalphas0):
        #     break
        logsumterms = np.concatenate((term0.reshape(-1,1), np.repeat(sumlogalphas0.reshape(1,-1),k,axis=0)),axis=1)
        # print(f'logsumterms shape is {logsumterms.shape}')
        logsumterms = scipy.special.logsumexp(logsumterms,axis=1).flatten()
        logalphas0 = logsumterms + np.log(gamma)
        # print(logalphas0)
        # for w in range(k):
        #     logsumterms = np.concatenate((term0[w,:],term1,term2))
        #     # logsumterms = np.concatenate((term0.reshape(-1,1), np.repeat(term1.reshape(1,-1),k,axis=0), np.repeat(term2.reshape(1,-1),k,axis=0)))
        #     logalphas[w]=logsumexp(logsumterms,b=[1]*(k+1)+[-1]*k)+np.log(gamma[w])
    return logalphas0


# @jit(nopython=True)
def jointH_long(logtheta,S,ref_order,maxperm=20):
    '''
    Learn the error parameter theta
    '''
    # perm = np.minimum(maxperm,int(np.ceil(0.2*K)))
    logpi = 0
    for k,_ in enumerate(ref_order):
        if k >= len(ref_order)-2:
            break
        alphas0 = init_alpha(ref_order,k+1, np.exp(logtheta))
        # print('alphas0')
        # print(alphas0)
        # alpha_k = fill(theta, alpha0[ref_order[:(k+1)]],0,S,k+1) # k is 0 based, so need to add by 1 
        logalpha_k = fill_norec_long_nb(logtheta, alphas0, S, k+1, order) # fill_norec_long(logtheta, alphas0, M, k,ref_order)
        # print(logalpha_k)
        # logpi_k = -np.log10(np.sum(alpha_k))
        logpi_k = -logsumexp(logalpha_k,b=[1]*len(logalpha_k))/np.log(10)
        print(logpi_k)
        # logpi_k = -np.log10(np.sum(np.exp(logalpha_k))) # -factor*np.log10(np.exp(0.1))
        logpi+=logpi_k
    return logpi



@jit(nopython=True)
def jointH(logtheta,S,ref_order):
    '''
    Learn the error parameter theta
    Args:
        logtheta: the log of the mutation rate
        S: number of snps
        ref_order: the order of computing the reference haplotypes
    Return:
        Negative log likelihood of P(h1,...,hk)
    '''
    # perm = np.minimum(maxperm,int(np.ceil(0.2*K)))
    logpi = 0
    theta = np.exp(logtheta)
    for k,_ in enumerate(ref_order):
        if k > len(ref_order)-2:
            break
        alphas0 = init_alpha(ref_order,k+1, theta)
        alpha_k = fill_norec(theta, alphas0, S, k+1,ref_order) # non recursive call
        logpi_k = -np.log10(np.sum(alpha_k))
        print('logpi_k is: ')
        print(k, ': ', logpi_k)
        logpi+=logpi_k
    return logpi


def jointH_test(theta,S,K,testhap):
    '''
    Learn the error parameter theta
    Args:
        logtheta: the log value of theta 
        S: number of snps used
        K: number of reference haplotypes
        testhap: the log likelihood of testing haplotype
    Return:
        Negative log likelihood of P(h_test| h_trains)
    '''
    # perm = np.minimum(maxperm,int(np.ceil(0.2*K)))
    logpi = 0
    order = np.arange(K)
    logtheta = np.log(theta)
    alphas0 = init_alpha(order,K, theta,testsnp=testhap[0])
    # print(f'alphas0: {alphas0}')
    alpha_k = fill_norec(theta, alphas0, S, K, order,testhap=testhap) # non recursive call
    # print(alpha_k)
    # logpi_k = -scipy.special(logalpha_k,b=[1]*(k+1))/np.log(10)
    # print(f'alpha_k: {alpha_k}')
    logpi_k = -np.log10(np.sum(alpha_k))
    logpi+=logpi_k
    return logpi


def jointH_test_log(theta,S,K,testhap):
    '''
    Learn the error parameter theta
    Args:
        logtheta: the log value of theta 
        S: number of snps used
        K: number of reference haplotypes
        testhap: the log likelihood of testing haplotype
    Return:
        Negative log likelihood of P(h_test| h_trains)
    '''
    # perm = np.minimum(maxperm,int(np.ceil(0.2*K)))
    logpi = 0
    order = np.arange(K)
    logtheta = np.log(theta)
    # theta = np.exp(logtheta)
    alphas0 = init_alpha(order,K, theta,testsnp=testhap[0])
    # print(f'alphas0 long: {alphas0}')
    logalpha_k = fill_norec_long(logtheta, alphas0, S, K, order,testhap=testhap) # fill_norec_long(logtheta, alphas0, M, k,ref_order)
    # print(logalpha_k)
    # logpi_k = -np.log10(np.sum(alpha_k))
    # print(f'long alpha_k is {np.exp(logalpha_k)}')
    logpi_k = -scipy.special.logsumexp(logalpha_k)/np.log(10)
    logpi+=logpi_k
    return logpi

# @jit(nopython=True)
def fill_norec(theta, alphas0, M, k,ref_order,testhap=np.array([])):
    '''
    j: Initial value should be 0
    recursively solving alpha_{j+1} 
    '''
    # alphas = np.zeros(k)
    # logalphas0 = np.log2(alphas0)
    # for i in range(k):
    # print(f'fill number of ref is {k}')
    # print(f'ref length is {len(ref_order)}')
    # print(f'ref end is {ref_order[k]}')
    for j in range(1,M):
        if testhap.size >0:
            hs = testhap[j]
        else:
            hs = H[ref_order[k],j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[ref_order[:k],j] 
        gamma = gammax(theta,hs,h_origs,k)
        # print(f'gamma is {gamma}')
        pj = np.exp(-(GM[j]-GM[j-1])/k)
        alphas = gamma*(pj*alphas0 + (1-pj)*np.sum(alphas0)/k) 
        # if j < 10:
        #     print('alphas')
        #     print(alphas)
        alphas0 = alphas
        # logalphas0 = logalphas
    return alphas

# @jit(nopython=True)
def init_alpha(ref_order,k, theta, testsnp=-1):
    '''
    This function compute the probability of P(h=k+1, X1=k); 
    or the joint probability of having k+1 reference haplotype and observing snps in haplotype k in the first locus
    alpha is a K vector, depending on K
    P(h=1, X1=1) = P(h=k+1| X1=j) * P(X1=j) = gammax(theta, H[ref_order[0],j],h_origs,j)*1/K
    '''
    if testsnp==-1:
        testsnp = H[ref_order[k],0]

    alphas0 = gammax(theta, testsnp,H[ref_order[:k],0],k)*1/k
    return alphas0



### python version ### 
def gammax_p(theta, hs,h_origs,k):
    '''
    Compute the P(h_{k+1, j+1}| X_{j+1}=x,h1,..,h_k)
    Args;
        h: value of h_{k+1, j}
        h_orig: value of h given it comes from jth reference
        theta: error rate need to be estimated
    '''
    
    boolean_array = hs==h_origs
    intercept = np.ones(k)
    
    return intercept*0.5*theta/(k+theta)+boolean_array*k/(k+theta)


def fill_norec_long_p(theta, alphas0, M, k,ref_order):
    '''
    j: Initial value should be 0
    recursively solving alpha_{j+1}
    Add more precision
    factor: a log based constant
    '''
    logalphas0 = np.log(alphas0)
    logalphas = np.zeros(k)
    for j in range(M-1):
        # j means j+1
        hs = H[ref_order[k],j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[ref_order[:k],j] 
        gamma = gammax_p(theta,hs,h_origs,k) # times a constant factor to avoid underflow
        # pj = np.exp(-(GM[j+1]-GM[j])/k)
        # log sum of exp(-(GM[j+1]-GM[j])/k + logalphas0) + sum (np.exp(logalphas0))
        # term0: -gm/k + logalphas0; 1 x k
        term0=(-(GM[j+1]-GM[j])/k+logalphas0).reshape(-1,1)
        # term1: (logalpha_i(x') - logk); 1 x k => need to be broadcast
        term1=(logalphas0-np.log(k)).reshape(1,-1) # k terms, plus sign
        # add12 = np.logaddexp(term0, term1)
        # term2: logalpha_i(x') - logk - gm/k; 1 x k => need to be broadcast to k by k
        term2=(logalphas0-np.log(k)-(GM[j+1]-GM[j])/k).reshape(1,-1) # k terms, minus sign
        
        # logsumterms = np.concatenate((term0,term1,term2))
        logsumterms = np.concatenate((term0.reshape(-1,1), np.repeat(term1.reshape(1,-1),k,axis=0), np.repeat(term2.reshape(1,-1),k,axis=0)),axis=1)
        logalphas=scipy.special.logsumexp(logsumterms,b=[1]*(k+1)+[-1]*k,axis=1)+np.log(gamma)
        # print(logalphas)
        # logalphas = np.log(gamma)+ np.log(pj*alphas0 + (1-pj)*np.sum(alphas0)/k)
        
        # addterm = [-(GM[j+1]-GM[j])/k+logalphas0+log(k-1), 
        # logalphas = np.log(gamma)+np.log(k*pj*np.exp(logalphas0) + (1-pj)*np.sum(np.exp(logalphas0))) - np.log(k) 
        # if j <= 20:
        #     print(logalphas)
        logalphas0 = logalphas
    return logalphas0

def jointH_long_p(logtheta,S,ref_order,maxperm=20):
    '''
    Learn the error parameter theta
    '''
    # perm = np.minimum(maxperm,int(np.ceil(0.2*K)))
    logpi = 0

    for k,_ in enumerate(ref_order):
        if k >= len(ref_order)-1:
            break
        alphas0 = init_alpha_p(ref_order,k+1, np.exp(logtheta))
        # print('alphas0')
        # print(alphas0)
        # alpha_k = fill(theta, alpha0[ref_order[:(k+1)]],0,S,k+1) # k is 0 based, so need to add by 1 
        logalpha_k = fill_norec_long_p(np.exp(logtheta), alphas0, S, k+1,ref_order) # non recursive call; this function would cause underflow
        # print(logalpha_k)
        logpi_k = -np.log10(np.sum(np.exp(logalpha_k))) # -factor*np.log10(np.exp(0.1))
        logpi+=logpi_k
    return logpi

def init_alpha_p(ref_order,k, theta):
    '''
    This function compute the probability of P(h=k+1, X1=k); 
    or the joint probability of having k+1 reference haplotype and observing snps in haplotype k in the first locus
    alpha is a K vector, depending on K
    P(h=k+1, X1=k) = P(h=k+1| X1=j) * P(X1=j) = gammax(theta, H[ref_order[0],j],h_origs,j)*1/K
    '''
    alphas0 = gammax(theta, H[ref_order[k],0],H[ref_order[:k],0],k)*1/k
    return alphas0


#### python end #####



@jit(nopython=True)
def forward(j,theta,K, testsnp):
    '''
    Description: 
        Computing P(X_j, h_{1,j})
    Args:
        j: position of the imputed genotype
        K: number of reference genotypes
        testsnp: snps vector of given individual
    Return:
        an array of forward alpha
    '''
    order = np.arange(K)
    alphas0 = init_alpha(order,K, theta,testsnp=testsnp[0])
    alpha_j = forfill(theta, alphas0, j, K, testsnp) # forfill(theta,alphas0,j,k,testsnp)
    return alpha_j

@jit(nopython=True)
def backward(M,j,K,theta,testsnp):
    '''
    Describ
        Compute P(h_{j,M}| x_{j-1})
    Args: 
        j, theta, L
        beta0: Base case, should be all 1s 

    '''
    beta0 = np.ones(K)
    beta_j = backfill(theta,beta0,M,j,K,testsnp) # backfill(theta,betaM,M,j,K,testsnp)
    return beta_j

@jit(nopython=True)
def backward_log(M,j,K,theta,testsnp):
    '''
    Describ
        Compute P(h_{j,M}| x_{j-1})
    Args: 
        j, theta, L
        beta0: Base case, should be all 1s 

    '''
    logbeta0 = np.zeros(K)
    beta_j = backfill_log(theta,logbeta0,M,j,K,testsnp) # backfill(theta,betaM,M,j,K,testsnp)
    return beta_j

@jit(nopython=True)
def backfill(theta,betaM,M,j,K,testsnp):
    for j in range(M-1,j-1,-1):
        hs = testsnp[j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[:K,j] 
        gamma = gammax(theta,hs,h_origs,K)
        pj =  np.exp(-(GM[j]-GM[j-1])/K)
        beta_j = pj*betaM*gamma + (1-pj)*np.sum(betaM*gamma)/K
        betaM =beta_j
    return betaM

@jit(nopython=True)
def backfill_log(theta,logbetaM,M,j,K,testsnp):
    for j in range(M-1,j-1,-1):
        hs = testsnp[j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[:K,j] 
        loggamma = np.log(gammax(theta,hs,h_origs,K))
        pj =  np.exp(-(GM[j]-GM[j-1])/K)
        logbeta_j = np.log(pj)+np.log(betaM)+loggamma + np.log((1-pj))+ np.log(np.sum(np.exp(logbetaM + loggamma))) - np.log(K)
        logbetaM =logbeta_j
    return logbetaM


@jit(nopython=True)
def forfill(theta,alphas0,M,k,testsnp):
    # testsnp is a vector right now
    for j in range(1,M):
        hs = testsnp[j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[:k,j] 
        gamma = gammax(theta,hs,h_origs,k)
        # print('gamma is: ')
        # print(gamma)
        # d = Distlist[j+1] - Distlist[j] # Distlist records the physical distance at each loci
        # pj = np.exp(-rhos[j-1]*d/k)
        pj = np.exp(-(GM[j]-GM[j-1])/k)
        alphas = gamma*(pj*alphas0 + (1-pj)*np.sum(alphas0)/k) 
        # if j < 10:
        #     print('alphas')
        #     print(alphas)
        alphas0 = alphas
        # logalphas0 = logalphas
    return alphas0

@jit(nopython=True)
def forfill_log(theta,alphas0,M,k,testsnp):
    # testsnp is a vector right now
    for j in range(1,M):
        hs = testsnp[j] # H is a k by S matrix with reference haplotypes; hs is h x j
        h_origs = H[:k,j] 
        loggamma = np.log(gammax(theta,hs,h_origs,k))
        # print('gamma is: ')
        # print(gamma)
        # d = Distlist[j+1] - Distlist[j] # Distlist records the physical distance at each loci
        # pj = np.exp(-rhos[j-1]*d/k)
        pj = np.exp(-(GM[j]-GM[j-1])/k)
        alphas = gamma*(pj*alphas0 + (1-pj)*np.sum(alphas0)/k) 
        # if j < 10:
        #     print('alphas')
        #     print(alphas)
        alphas0 = alphas
        # logalphas0 = logalphas
    return alphas0


def posterior(alphabeta, j,  K, theta):
    # compute the posterior probability of P(h_j=1| h_{1,M})
    marginal_vector1 = alphabeta*(H[:,j]==1)
    marginal_vector1 = marginal_vector1*(K/(K+theta) + 0.5*theta/(K+theta))
    marginal_vector2 = alphabeta*(H[:,j]==0)
    marginal_vector2 = marginal_vector2*(0.5*theta/(K+theta))

    return 1/(1+np.sum(marginal_vector2)/np.sum(marginal_vector1))

    # return np.sum(marginal_vector1)/(np.sum(marginal_vector1)+np.sum(marginal_vector2))

def logposterior(logalphabeta, j,  K, theta):
    # compute the posterior probability of P(h_j=1| h_{1,M})
    marginal_vector1 = logalphabeta*(H[:,j]==1)
    marginal_vector1 = marginal_vector1 + np.log(K/(K+theta) + 0.5*theta/(K+theta))
    marginal_vector2 = logalphabeta + np.log(H[:,j]==0)
    marginal_vector2 = marginal_vector2 + np.log(0.5*theta/(K+theta))

    return 1/(1+np.sum(np.exp(marginal_vector2))/np.sum(np.exp(marginal_vector1)))

    # return np.sum(marginal_vector1)/(np.sum(marginal_vector1)+np.sum(marginal_vector2))





def permuteHap(order, seed=0):
    if seed>0:
        np.random.seed(seed)

    # global ref_order
    np.random.shuffle(order)
    return


# def genetic_map(ref, dist, snps):
#     '''
#     Args: 
#         ref: a vector of snps position
#         dist: a vector of genetic mapping value, 1-1 as ref
#         snps: a vector of snps of testing
#     step 1: find the index of snps which has records in ref
#     step 2: get the corresponding distance of each records
#     step 3: impute the genetic score for each of the missing records
#     ''' 
#     refindices=np.in1d(ref,snps)
#     testindices=np.in1d(snps,ref)

    



# Simulation start
# K = 10
# S = 10000

# # Distlist = np.loadtxt('example.dist')
# Distlist = np.arange(S)
# # rhos = np.loadtxt('example.rho')
# rhos = np.random.uniform(0.1,0.2,S)
# # H = np.loadt('example.hap')
# H = np.random.binomial(1,0.5, (K,S))
# print(H)


# datapath='/home/boyang1995/research/Li-Stephens/PrivateGenomes.jl/data/'
# mapfile='10k_SNP.map'
# gene_df = pd.read_csv(f'{datapath}{mapfile}',sep=' ')
# GM = gene_df.infer_map.values  # genetic mapping


# alpha0 = np.random.uniform(0,0.1,K)
# alpha0 = alpha0/np.sum(alpha0)
# logtheta = np.log(1/(np.sum(1/np.arange(1,K+1))))

# jointHest = jointH(theta,alpha0,S,K)
# print(jointHest)
# alpha_t = fill(theta, alpha0,0,S,K)
# print(alpha_t)
# start = time.time()
# # res = minimize(jointH, logtheta, method='nelder-mead', args=(alpha0,S,K,1))
# order = np.random.choice(K,K,replace=False)
# # res = minimize(jointH, logtheta, method='nelder-mead', args=(alpha0,S,K,),tol=1)
# # res = scipy.optimize.brent(jointH, args=(alpha0,S,K,),tol=1)
# res = scipy.optimize.minimize_scalar(jointH,args=(alpha0,S,order),tol=1)
# end = time.time()
# print(f'execution time is {end-start}')
# print(res)

# start = time.time()
# # res = minimize(jointH, logtheta, method='nelder-mead', args=(alpha0,S,K,1))
# # del ref_order
# order = np.random.choice(K,K,replace=False)
# print(f'order is {order}')
# # res = minimize(jointH, logtheta, method='nelder-mead', args=(alpha0,S,K,),tol=1)
# # res = scipy.optimize.brent(jointH, args=(alpha0,S,K,),tol=1)
# res = scipy.optimize.minimize_scalar(jointH_long,args=(S,order),tol=0.1)
# # res = scipy.optimize.minimize_scalar(jointH,args=(S,order),tol=0.1)
# end = time.time()
# print(f'execution time is {end-start}')
# print(res)

# Simulation end



####
# real analysis


# infpath='/home/boyang1995/research/Li-Stephens/PrivateGenomes.jl/orig_data/1000GP_Phase3/'
# infofile='genetic_map_chr15_combined_b37.txt'
datapath='/home/boyang1995/research/Li-Stephens/PrivateGenomes.jl/data/'
snpid='805_SNP.legend'
test='805_SNP_1000G_real.hapt.test'
train='805_SNP_1000G_real.hapt.train'
mapfile='805_SNP.map'

H = pd.read_csv(f'{datapath}{train}',sep=' ',header=None).iloc[:,2:].values
gene_df = pd.read_csv(f'{datapath}{mapfile}',sep=' ')
GM = gene_df.infer_map.values  # genetic mapping

H_test_df = pd.read_csv(f'{datapath}{test}',sep=' ',header=None)
H_test = H_test_df.iloc[:,2:].values
# H_test_lab = H_test.iloc[:,:2].values

K = H.shape[0]
S = H.shape[1]
# S = 5000


impute_FLAG=False
test_loglike=True
theta_learning=False
theta = np.exp(-1)
N = H_test.shape[0]
if impute_FLAG:
    '''
    Pearson correlation, mse error, MAF
    methods: 
    1. HMM
    2. majority vote (MSE only)
    3. random guess 
    '''
    print('############# imputation test ############')

    print(f'K is {K}')
    print(f'S is {S}')
    # N = 20
    rhos_hmm = []
    # rhos_hard_hmm = []
    rhos_rg = [] # random guess
    mses_hmm = []
    mses_hmm_hard = []
    # mse_rg = []
    
    MAFlist = []
    mse_MAF = []
    # j = 40
    Indices = []
    for j in range(0, 500):
        print(f'current snp is {j}')
    # for j in range(4):
        Indices.append(j)
        best_ks = []
        postlist = []
        for i in tqdm(range(N)):
            # -0.06
            alpha_j = forward(j,theta, K, H_test[i]) # forward(j,theta,K, testsnp)
            assert np.all(np.isfinite(alpha_j))
            # print(f'Forward prob: ')
            # print(alpha_j)
            beta_j = backward(S, j+1, K, theta, H_test[i]) # backward(M,j,K,theta,testsnp)
            assert np.all(np.isfinite(beta_j))
            # print(f'Backward prob: ')
            # print(beta_j)
            # logalphabeta = logalpha_i + logbeta_j
            alphabeta = alpha_j*beta_j
            # best_k = np.argmax(alphabeta)
            # best_ks.append(best_k)
            # P = log_posterior(logalphabeta, j, K, theta)  # log_posterior(logalphabeta, j,  K, theta)
            P = posterior(alphabeta, j, K,theta)  # posterior(alphabeta, j,  K, theta)
            if np.isnan(P):
                print(alpha_j)
                print(beta_j)
                print(alphabeta)
            postlist.append(P)
        MAF = np.sum(H[:,j])/H.shape[0]
        # print(f'MAF is {MAF}, {1-MAF}')
        if MAF > (1-MAF):
            impute = 1 
        else:
            impute = 0
        MAF = np.amin((MAF,1-MAF))
        MAFlist.append(MAF)
        mse_MAF.append(((impute- H_test[:N,j])**2).mean())
        # print(postlist)
        hard_imputes = np.round(postlist)
        postlist = np.array(postlist)
        # print(postlist)

        rho_P = pearsonr(postlist,H_test[:N,j])[0]
        rho_sanity = pearsonr(H[np.random.choice(K,N),j],H_test[:N,j])[0]
        mse_hmm_hard = ((hard_imputes- H_test[:N,j])**2).mean()
        mse_hmm = ((postlist- H_test[:N,j])**2).mean()

        rhos_hmm.append(rho_P)
        rhos_rg.append(rho_sanity)
        mses_hmm_hard.append(mse_hmm_hard)
        mses_hmm.append(mse_hmm)

        print(f'Pearson correlation for index {j} posterior is {rho_P}')
        print(f'Random pearson correlation is {rho_sanity}')
    # postlist = np.array(postlist).reshape(-1,1)
        # MAFlist = np.array(MAFlist).reshape(-1,1)
        # rhos_hmm = np.array(rhos_hmm).reshape(-1,1)
        # # rho_sanity = np.array(rho_sanity).reshape(-1,1)
        # rhos_rg = np.array(rhos_rg).reshape(-1,1)
        # mses_hmm = np.array(mses_hmm).reshape(-1,1)
        # mses_hmm_hard = np.array(mses_hmm_hard).reshape(-1,1)
        # MAFlist = np.array(MAFlist).reshape(-1,1)
        # Indices = np.array(Indices).reshape(-1,1)
        # mse_MAF = np.array(mse_MAF).reshape(-1,1)
        
        summary = np.concatenate((np.array(Indices).reshape(-1,1), np.array(MAFlist).reshape(-1,1), np.array(rhos_hmm).reshape(-1,1), np.array(rhos_rg).reshape(-1,1),  np.array(mses_hmm).reshape(-1,1), np.array(mses_hmm_hard).reshape(-1,1), np.array(mse_MAF).reshape(-1,1)),axis=1)
        summary_df = pd.DataFrame(data=summary, columns=['Index', 'MAF', 'r2_hmm', 'r2_rand', 'mse_hmm', 'mse_hmm_hard', 'mse_MAF'])
        summary_df.to_csv('summary.txt',sep=' ',index=False)

        
        # best_ks = np.array(best_ks)
        # rho = pearsonr(H[best_ks,j],H_test[:N,j])[0]
        
        # print(f'Pearson correlation for index {j} is {rho}')
        

       
        

        

    # np.savetxt('impute.txt',H[best_ks,j])
    # np.savetxt('actual.txt',H_test[:N,j])
   
    
    # print(f'Imputation error is {np.sum(np.abs(H_test[:,j]-impute))/H_test.shape[0]}')

if test_loglike:
    loglikes = []
    indindex = []
    GM = GM[1:] - GM[:-1]
    GM = GM[GM>=0]
    Genetic_dist = np.mean(GM)
    GM = np.repeat(Genetic_dist, S)
    
    # S = 100
    print('############# test data log likelihood computation ############')
    print(f'Total tested snps are {S}')
    print(f'Total samples are {K}')
    for i in range(H_test.shape[0]):
    # for i in [255,257,370,371,464,491,773,811,820,822,835]:
        indindex.append(i)
        # loglike = jointH_test(theta, S, K, testhap=H_test[i]) # jointH_test(logtheta,S,K,testhap)
        # print(f'no precision enhance: {loglike}')
        loglike = jointH_test_log(theta, S, K, testhap=H_test[i]) # jointH_test(logtheta,S,K,testhap)
        print(f'precision enhance: {loglike}')
        loglikes.append(loglike)
        # print(f'neg log lik for ind {i} is {loglike}')
    loglikes = np.array(loglikes).reshape(-1,1)
    indindex = np.array(indindex).reshape(-1,1)
    logsummary = np.concatenate((indindex,loglikes),axis=1)
    # print(loglikes)
    np.savetxt('loglike_805.txt',logsummary,fmt='%i %.18e')
    print('############# test data log likelihood computation done ############')


if theta_learning:

    xs = []
    for i in range(10):
        Ks = 500
        order = np.random.choice(K,Ks,replace=False)
        
        S = 1000
        # order = np.arange(K)
        # order = np.array([1,2,3,4,1])
        
        # H = np.random.binomial(1,0.5, (K,S))
        # alpha0 = np.random.uniform(0,0.1,K)
        # alpha0 = alpha0/np.sum(alpha0)
        start = time.time()
        # logpi = jointH(-1,S,order)
        # print(logpi)
        res = scipy.optimize.minimize_scalar(jointH,args=(S,order))
        xs.append(np.exp(res.x))
    xs = np.array(xs)
    print(xs)
    print(f'mean value is {np.mean(xs)}, std is {np.std(xs)}')

    # res2 = scipy.optimize.minimize_scalar(jointH_long,args=(S,order),bounds=(-50,0),method='bounded')
    # print(res2)
    # res3 = scipy.optimize.minimize_scalar(jointH_long_p,args=(S,order),tol=0.1)
    # print(res3)
    end = time.time()
    print(f'execution time is {end-start}')





