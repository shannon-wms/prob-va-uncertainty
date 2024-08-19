"""
Script containing various utility functions.

Author: Shannon Williams

Date: 25/09/2023
"""

import os
import numpy as np
from scipy.stats import t, rv_continuous

def mixture_dist(x: float, locs: list, scales: list, distribution: rv_continuous = t, 
                 weights: list = None, **kwargs):
    """Evaluate cdf of a weighted mixture of location-scale distributions.

    Args:
        x (float): Point x at which to evaluate cdf.
        locs (list): List of location parameters.
        scales (float): Scale parameter if constant across distributions, or list of these.
        distribution (rv_continuous, optional): Distribution to evaluate. Defaults to t.
        weights (list, optional): List of weights. If none, assumes an equal weighting of disributions. Defaults to None.

    Returns:
        float: Weighted average of t-distributions evaluated at x.
    """
    
    if not isinstance(scales, list):
        scales = [scales for _ in range(len(locs))]
    
    # Assign equal weights
    if weights is None:
        weights = [1/len(locs) for _ in range(len(locs))]
    
    dx = [] # Evaluate cdf at x for each mixture component
    for i in range(len(locs)):
        dx.append(distribution.cdf(x, loc = locs[i], scale = scales[i], **kwargs))
    
    # Return weighted average of cdf
    return np.average(np.vstack(dx), axis = 0, weights = weights)

def _validate_interval(fun, x0: float, x1: float, 
                       quantile: float = 0.0, **kwargs):
    fx0 = fun(x0, **kwargs) - quantile    
    fx1 = fun(x1, **kwargs) - quantile    

    return fx0 > 0, fx1 > 0

def bisection(interval: list, fun, c: float = 0.0, 
              n_iter: int = 10, tol: float = 1E-2, **kwargs):
    """Simple implementation of bisection algorithm to find root x of an 
    expression fun(x)=c.

    Args:
        interval (list): List specifying the endpoints of the estimated interval containing the root.
        fun (function): Function to be evaluated.
        c (float, optional): RHS of expression fun(x)=c to be solved. Defaults to 0.0.
        n_iter (int, optional): Maximum number of iterations of algorithm. Defaults to 10.
        tol (float, optional): Tolerance of accepting solution. Defaults to 1E-2.

    Returns:
        float: x solving fun(x)=c.
    """
    x0, x1 = interval[0], interval[1]
    
    val = _validate_interval(fun, x0, x1, c, **kwargs)
    # f(x0) positive so the solution is less than x0
    if val[0]: 
        return x0 - 1 
    # f(x1) negative so the solution is greater than x1
    if not val[1]:
        return x1 + 1
    
    for _ in range(n_iter):
        rt = x0 + ((x1 - x0) / 2)
        y = fun(rt, **kwargs) - c
        
        # Accept if within tolerance
        if np.abs(y) < tol:
            return rt
        
        # Check the interval and shrink according to which side solution is on
        val = _validate_interval(fun, x0, rt, c, **kwargs)
        
        # f(x0) * f(x1) > 0
        if all(val) or not any(val):
            x0 = rt
        else: # f(x0) * f(x1) < 0
            x1 = rt
            
    print("Convergence not reached. Increase n_iter.")
    return rt

class Ivespa:

    def __init__(self):
        self.beta_vec = np.array([2.82541738, 3.54169211])
        self.sigma2 = 0.5896221362244476
        self.matV = np.array([[ 0.07143395, -0.07353302], [-0.07353302,  0.08482847]])
        self.df = 128    
    
def set_ivespa_obs(heights: list, volcano_height: float = 0.0):
    """Initialise Bayesian model with IVESPA data and set observations to obtain parameters.

    Args:
        heights (list): List of plume heights, in m a.s.l.
        volcano_height (float, optional): Vent height, in m a.s.l. Defaults to 
        0.0.
    """
    if not isinstance(heights, list):
        heights = [heights]
    
    # convert from m to km asl
    heights_km_avl = [(h - volcano_height) / 1000 for h in heights]

    Xp = np.ones([np.size(heights_km_avl), 2]) 
    Xp[:, 1] = heights_km_avl

    ivespa = Ivespa()

    ivespa.mu = np.matmul(Xp, ivespa.beta_vec)
    Id = np.identity(np.size(heights_km_avl), dtype = np.float64) 
    Sigma2 = ivespa.sigma2 * (Id + np.matmul(np.matmul(Xp, ivespa.matV), Xp.T)) 
    ivespa.Sigma = np.sqrt(np.diag(Sigma2))

    return ivespa

def sort_member_dirs(name_output_dir: str):
    """Sort NAME ensemble member output directories into ascending order.

    Args:
        name_output_dir (str): Directory containing NAME ensemble outputs.

    Returns:
        list: Sorted list of ensemble member directory paths.
    """
    # Get member directories
    member_dirs =  [f.path + "/" for f in os.scandir(name_output_dir) 
                    if f.is_dir() and f.name.startswith("member")]

    # Sort the directories
    sorted_inds = list(np.argsort(
        [int(member_dirs[i].split("/")[-2].split("_")[-1]) 
         for i in range(len(member_dirs))]
        ))

    # Return sorted directories
    member_dirs = [member_dirs[i] for i in sorted_inds]
    return member_dirs
    