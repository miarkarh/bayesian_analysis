"""
Making parameter vectors.

@author: Mikko Karhunen
"""

import numpy as np
from latin_hyper_cube import latin_hyper_cube as lhc

# # names = "# m_c, C, lambda_g, A_g"
# # limits = np.array([[1.1, 1.8], [0.5, 10], [0.01, 0.5], [0.2, 10]])
# sigma0_limits = [10, 30]  # mb
# lambda_limits = [0.05, 0.5]
# Qs02_limits = [0.1, 5]
# limits = np.array([sigma0_limits, lambda_limits, Qs02_limits])
# # params = np.random.uniform(low=limits[:, 0], high=limits[:, 1], size=(200, 4)) # lhc(400, 4, limits, 1)
# params = lhc(200, 3, limits)

# np.savetxt("data/training data/parameters/lhc_samples_200_noC.dat", params, fmt='%.10e')
