import ot
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm

def wasserstein_gaussian_square(mu1, sigma1, mu2, sigma2):
    """
    Compute the 2-Wasserstein distance between two Gaussian distributions.

    Parameters:
    - mu1, mu2: Mean vectors (shape: [d])
    - sigma1, sigma2: Covariance matrices (shape: [d, d])

    Returns:
    - W2_distance: The 2-Wasserstein distance
    """
    # Ensure inputs are arrays
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Compute the squared Euclidean distance between the means
    mean_diff = mu1 - mu2
    mean_norm = np.dot(mean_diff, mean_diff)

    # Compute the matrix square roots
    sigma1_sqrt = sqrtm(sigma1)

    # Compute the product sigma1_sqrt * sigma2 * sigma1_sqrt
    sigma1_sqrt_sigma2_sigma1_sqrt = sigma1_sqrt @ sigma2 @ sigma1_sqrt

    # Compute the square root of the product matrix
    cov_mean = sqrtm(sigma1_sqrt_sigma2_sigma1_sqrt)

    # Handle numerical errors (if any imaginary parts due to computation)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Compute the trace term
    trace_term = np.trace(sigma1 + sigma2 - 2 * cov_mean)

    # Compute the squared Wasserstein-2 distance
    W2_squared = mean_norm + trace_term

    return W2_squared

def W2_diff(Y, Y1, Y2):
    num_data = Y.shape[0]
    weights = np.ones(num_data) / num_data
    weights1 = np.ones(num_data) / num_data
    weights2 = np.ones(num_data) / num_data

    M1 = ot.dist(Y, Y1, metric='euclidean') ** 2
    M2 = ot.dist(Y, Y2, metric='euclidean') ** 2

    transport_plan1 = ot.emd(weights, weights1, M1)
    W2_distance_squared1 = np.sum(transport_plan1 * M1)

    transport_plan2 = ot.emd(weights, weights2, M2)
    W2_distance_squared2 = np.sum(transport_plan2 * M2)

    result = W2_distance_squared2 - W2_distance_squared1
    return result

def kl_divergence_knn(X, Y, k=4):
    n, d = X.shape
    m, _ = Y.shape

    knn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = knn.kneighbors(X)
    rho = distances[:, -1]

    knn = NearestNeighbors(n_neighbors=k).fit(Y)
    distances, _ = knn.kneighbors(X)
    nu = distances[:, -1]
    # print(d, m, n, nu, rho)

    return np.log(m / (n - 1)) + d * np.log(nu / rho).mean()


def kl_mvn(to, fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    m_to, S_to = to
    m_fr, S_fr = fr
    
    d = m_fr - m_to
    
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.


def solve_for_B(alpha=0.1, Delta=0.0, t = 0.0):
    B_low = max(np.floor(np.emath.logn(2 + 2*t, (2 + 2*t)/alpha )), np.floor(np.emath.logn((2 + 2*t)/(1 + 2*Delta), (1 + t)/alpha )))
    B_up = np.ceil(np.emath.logn((2 + 2*t)/(1 + 2*Delta), (2 + 2*t)/alpha))
    def function_Q(B):
        return ((1/2 - Delta)**B + (1/2 + Delta)**B)*(1 + t)**(-B + 1)
    for B in range(int(B_low), int(B_up) + 1):
        if function_Q(B) <= alpha:
            break
    return B