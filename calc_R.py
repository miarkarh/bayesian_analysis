# -*- coding: utf-8 -*-
"""
Calculate nucleus proton F2/F2/A ratio in puck cluster computer.

@author: Mikko Karhunen
"""

import sys
import numpy as np
import physics

i = int(sys.argv[1])

Q2 = np.geomspace(1, 1000, 20)
x = 0.001
sigma0, lambd, Qs02 = np.loadtxt("data/other/light_posterior_hundred_samples.dat")[i]
ratio_nuc_proton = [physics.ratio_R(x, Q2, sigma0=sigma0, lambd=lambd, Q02=Qs02, A=197) for Q2 in Q2]
np.savetxt('ratio_' + str(i) + '.dat', ratio_nuc_proton, newline=' ')
# This is an example that can be used to pick hundred samples from posterior
# samples[np.random.randint(0, samples.shape[0], 100)]
