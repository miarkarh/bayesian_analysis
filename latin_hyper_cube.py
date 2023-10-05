"""
Latin hypercube from scipy.

@author: Mikko Artturi Karhunen
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube as lhc


def latin_hyper_cube(N, ndim, scale_limits=None, strenght=1):
    """
    Generate latin hypercube parameter vector samples from scipy.

    Parameters
    ----------
    N : int
        How many samples.
    ndim : int
        How many parameters.
    scale_limits : array, optional
        Limits for parameters. The default is None.
    strenght : int, optional
        Only 1 or 2. 2 if orthogonal latin hypercube is used from scipy. The default is 1.

    Returns
    -------
    lhc_samples : array
        Parameter vector samples sampled from latin hypercube.

    """
    lhc_sampler = lhc(ndim, strength=strenght)
    lhc_samples = lhc_sampler.random(N)
    if scale_limits is None: return lhc_samples
    # Scaling of samples to parameters
    for i in range(np.shape(lhc_samples)[1]):
        lhc_samples[:, i] *= np.abs(np.max(scale_limits[i]) - np.min(scale_limits[i]))
        lhc_samples[:, i] += np.min(scale_limits[i])
    return lhc_samples
