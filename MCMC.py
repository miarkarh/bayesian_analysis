"""
Markov chain Monte Carlo sampling with emcee.

@author: Mikko Artturi Karhunen
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt


def mcmc_sampling(par_limits, log_probability, nwalkers=200, nwalks=2000, burn=500, thin=1, flat=False, moves=None, plot_progress=True, walkers_par_labels=None):
    """
    Markov chain Monte Carlo sampling of log-probability function using emcee.

    Parameters
    ----------
    par_limits : array
        Limits for parameters. This limits walkers' starting locations. Also from it tells how many parameters there are.
    log_probability : function
        Log-probability function f(sample), which to sample.
    nwalkers : int, optional
        How many walkers used in sampling process. The default is 200.
    nwalks : int, optional
        How many steps walkers take after burn stage. The default is 2000.
    burn : int, optional
        How many steps taken in burn stage of sampling for walkers to find the distribution. The default is 500.
    thin : int, optional
        Take only every thin steps from the chain. The default is 1.
    flat : bool, optional
        Defines if walkers chains are combined into one chain. The default is False.
    moves : emcee move, optional
        What emcee move algorithm to use. The default is None, which means stretch move in emcee.
    plot_progress : bool, optional
        Defines if walkers sampling progress are plotted. The default is False.
    walkers_par_labels : array, optional
        Names for each parameter in order. The default is None.

    Returns
    -------
    samples : Array
        Chain of samples.

    """
    # MCMC sampling
    ndim = len(par_limits)  # How many parameters
    p0 = np.array([]).reshape(nwalkers, 0)
    # Walker initialization
    for limits in par_limits:
        p1 = np.random.rand(nwalkers) * np.abs(limits[1] - limits[0]) + np.min(limits)
        p0 = np.column_stack((p0, p1))

    mcmc = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=moves)
    # Burn stage
    if burn == 0: burn = 1
    state = mcmc.run_mcmc(p0, burn)
    mcmc.reset()
    # The sampling run
    mcmc.run_mcmc(state, nwalks, progress=True)

    #print(mcmc.get_autocorr_time(thin=1, discard=100))

    # Average how ofter samples were accepted in the process.
    print('Mean acceptance ratio:' + str(np.mean(mcmc.acceptance_fraction)))
    # Get and save into chain
    samples = mcmc.get_chain(thin=thin, flat=flat)

    if plot_progress:
        plot_sampling_progress(mcmc.get_chain(), ndim, walkers_par_labels)
    return samples


def plot_sampling_progress(samples, ndim, labels=None):
    """
    Plots sampling progress of walkers from unflattened chain.

    Parameters
    ----------
    samples : array
        Unflattened chain from emcee.
    ndim : int
        Number of parameters.
    labels : array, optional
        Names of parameters. The default is None.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(ndim, figsize=(10, 2), sharex=True)

    # Ensure axes is always iterable
    if ndim == 1:
        axes = [axes]

    if labels is None:
        labels = []
        for i in range(ndim):
            labels.append("")  # labels.append("par" + str(i + 1))
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig.savefig("mcmc_walkers.pdf")
