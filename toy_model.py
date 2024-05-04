# -*- coding: utf-8 -*-
"""
A simple toy model for testing.

@author: Mikko Artturi Karhunen
"""

import numpy as np
import PCA_emulator
import probability_formulas as prob
from latin_hyper_cube import latin_hyper_cube
from bayesian_analysis_functions import plotting
import MCMC
import matplotlib.pyplot as plt
from bayesian_analysis_functions import z_score


def toy_model(npc, whiten, n_restarts, zoom=None, only_z=False):
    """
    Make bayesian analysis for a toy model like linear function.

    This is for all kinds of testing with easy models.

    Parameters
    ----------
    npc : int
        How many first principal components are emulated.
    whiten : bool
        If whitening is done in pca transformation.
    n_restarts : int
        How many times emulator searches best fit to training data.
    zoom : string, optional
        Defines if distribution is zoomed to relevant region. The default is None, meaning the whole
        parameter space is shown.
    only_z : bool, optional
        If one wants to only calculate and see z-score.

    Returns
    -------
    None.

    """
    # model = lambda aa, bb, cc, xx : bb*np.log(aa*xx)+cc
    # model = lambda aa, bb, cc, xx: aa * xx**2 + bb * xx + cc
    model = lambda aa, bb, cc, xx: (aa + bb) * xx + cc
    training_samples = 300

    x = np.linspace(0.1, 100, 400)

    atru, btru, ctru = [7, 1.4, 2.2]
    exp = model(atru, btru, ctru, x) + 0.1 * np.random.normal(0, 1, x.shape) * model(atru, btru, ctru, x)
    exp_std = exp * 0.1

    par_limits = np.array([[1, 10], [1, 10], [1, 10]])
    a, b, c = latin_hyper_cube(training_samples, 3, par_limits, 1).T
    X = np.vstack((a, b, c)).T
    Y = np.array([model(*p, x) for p in X])

    gp = PCA_emulator.PCA_GPE(npc, whiten)
    gp.train(X, Y, par_limits, n_restarts=n_restarts)  # , consta_k1_bounds=[1e-10, 1e10],
    # length_scale_bounds=[0.01, 1000], noise_level_bounds=[1e-16, 1e5])

    plt.figure()
    plt.plot(np.arange(0, npc), np.array(1 - gp.pca.explained_variance_ratio_)[:npc])
    plt.show()
    emulator = lambda theta: gp.predict(theta, False, True)
    log_prob = lambda theta: prob.log_posterior(exp, emulator, theta, None, exp_std, par_limits)

    emu_std = lambda theta: gp.predict(theta, True, False)[1]
    sample_y = lambda theta: gp.sample_y(theta, 400)

    test_pars = np.random.uniform(*par_limits.T, (300, par_limits.shape[0]))
    modelp = np.array([model(*p, x) for p in test_pars])
    print(modelp.shape)
    z_score(sample_y(test_pars), emu_std(test_pars), modelp)
    if only_z: return
    # testing if mean predictions - real model are close to zero.
    # plt.figure()
    # ems = lambda theta: gp.predict(theta, False, False)
    # plt.hist(ems(test_pars)[:, :2] - modelp[:, :2])
    # plt.show()

    # samples = MCMC.mcmc_sampling(par_limits, log_prob, 100, 100, 100, 1, plot_progress=True, walkers_par_labels=["a", "b", "c"])
    # plotting(samples, par_limits, ["a", "b", "c"], zoom)


toy_model(3, True, 10, None, only_z=1)
