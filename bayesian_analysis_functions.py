"""
Here are some methods to help with coding many setups.

@author: Mikko Artturi Karhunen.
"""

# from plyer import notification
import numpy as np
import PCA_emulator as pca_emu
import MCMC
import matplotlib.pyplot as plt
import corner
import probability_formulas as prob_calc
import pickle


def plotting(samples, par_limits=None, labels=None, zoom=None, suptitle=None,
             save=False, fname=None, fig=None, ax=None):
    """
    Plot marginal distributions and correlations of parameters by histograms from samples.

    Code is a little messy.

    Parameters
    ----------
    samples : array or list of arrays
        The chain of parameters from MCMC. Can be a list of several chains for plotting multiple
        posteriors to the same figure.
    par_limits : array, optional
        Limits of the parameters in the posterior. The default is None.
    labels : array, optional
        Names of each parameter. The default is None.
    zoom : string, optional
        Defines if distribution is zoomed to relevant region. The default is None, meaning whole
        parameter space is shown. Other option is 'auto'.
    suptitle : string, optional
        Title for plot. The default is None.
    save : bool, optional
        If to save the figure. The default is False.
    fname : string, optional
        File name of saved figure. The default is None.
    fig : figure, optional
        If one wants to plot to some pre-existing figure. The default is None.
    ax : axes, optional
        Give also axes of the optional pre-existing figure. The default is None

    Raises
    ------
    Exception
        If zooming is tried with too few parameters.

    Returns
    -------
    figure, array, axises
        Returns the figure, the parameter limits and the axises of the figure.

    """
    if suptitle is not None: fig.suptitle(suptitle, y=0.95)

    # Automatically find relevant region.
    if zoom == 'auto':
        par_limits = find_region(samples)
    else: par_limits = zoom

    if not isinstance(samples, np.ndarray):
        nsamp = len(samples)
        colors = ("tab:" + c for c in ("blue", "red", "orange", "green", "purple", "pink", "gray"))
        figure = corner.corner(samples[0], labels=labels, bins=100, smooth=1.5, smooth1d=1,
                               plot_datapoints=False, range=par_limits, color=next(colors))
        for i in range(1, nsamp):
            figure = corner.corner(samples[i], bins=100, smooth=1.5, smooth1d=1,
                                   plot_datapoints=False, range=par_limits, fig=figure, color=next(colors))
    else:
        figure = corner.corner(samples, bins=100, labels=labels, smooth=1.5, smooth1d=1,
                               show_titles=True, plot_datapoints=False,
                               range=par_limits)

    # for axi in fig.get_axes():
    #     axi.label_outer()

    # Saving the figure
    if save:
        if fname is not None:
            figure.savefig(fname)
        else:
            figure.savefig('posterior_distribution.pdf')
    # if suptitle is not None and save: fig.savefig('suptitle+'.png')
    return figure, np.array(par_limits)


def find_region(samples):
    """
    Find automatically the relevant region for the posterior distribution.

    Parameters
    ----------
    samples : array or list of arrays
        The chain of parameters from MCMC. Can be a list of several chains from several posteriors.

    Returns
    -------
    ndarray
        Numpy array of parameter limits for relevant region.
    """
    if not isinstance(samples, np.ndarray):
        nsamp = len(samples)
        parlims = []
        for i in range(nsamp):
            parlims.append(find_region(samples[i]))

        # Find the minimum and maximum values element-wise across all arrays
        min_pars = np.min(parlims, axis=0)
        max_pars = np.max(parlims, axis=0)

        return np.vstack((min_pars[:, 0], max_pars[:, 1])).T
    nparameters = samples.shape[1]
    mean = []
    std = []
    for i in range(nparameters):
        m = np.percentile(samples[:, i], [16, 50, 84])
        mean.append(m[1])
        std.append((m[2] - m[0]) / 2)
    par_limits = []
    # Change for wider distribution region.
    wide = 3
    for i in range(nparameters):
        par_limits += [[mean[i] - wide * std[i], mean[i] + wide * std[i]]]
    return np.array(par_limits)


def z_score(pred_samps, pred_std, true_model, zoom=False, save_fig=False,
            fname='z_score.png', pred_mean=None, title=''):
    """
    Generate a z-score histogram.

    Parameters
    ----------
    pred : array
        The sampled predictions from emulator.
    pred_std : array
        Standard deviations of predictions.
    true_model : array
        True model data calculated with same parametisations as which were used in predictions. The
        model used to generate data should be same as which emulator was trained with.
    zoom : bool, optional
        If z-score is to be zoomed. This could be useful, if results are bad or weird.
        The default is False.
    save_fig : bool, optional
        If z-score is to be saved. The default is False.
    fname : string, optional
        Name of the saved plot file, if save_fig is True. The default is 'z_score.png'.
    pred_mean : array
        The used predicted values from emulator. For example returned means from emulator.

    Returns
    -------
    None.

    """
    if pred_mean is None: pred_mean = np.mean(pred_samps, axis=1)
    emu_error_perc = np.mean(np.abs((pred_mean - true_model) / true_model) * 100)
    print(np.max(np.abs((pred_mean - true_model) / pred_mean) * 100))

    print("<(emu - model) / emu> percent:", emu_error_perc)

    pred_samps = np.mean(pred_samps, axis=1)
    z = [(pred_samps[:, :] - true_model) / pred_std]
    # for i in range(pred_samps.shape[1])]
    z = np.ravel(z)
    mean = np.mean(z)
    std = np.std(z)

    # emu_error_perc = np.mean(np.abs(pred_std / np.mean(pred, axis=1) * 100))
    # print("mean of abs emulator stds (percent):", emu_error_perc)

    # plt.figure(figsize=(4, 4))
    # xx = np.linspace(0, np.max(pred_mean), 50)
    # yy = xx
    # plt.plot(xx, yy, color="orange")
    # plt.scatter(pred_mean, true_model, s=1)
    # plt.show()

    # plotting
    plt.figure(figsize=(4, 4))
    if zoom: rang = [-10, 10]
    else: rang = None
    plt.hist(z, 500, range=rang, density=True)  # Plot mean as a vertical line

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=0.9, alpha=0.8)
    plt.axvline(mean + std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    plt.axvline(mean - std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    if title != '': title = title + '\n'
    plt.title(title + ('Mean: {:.3f}, Std: {:.3f} \n').format(mean, std) + r"$\Delta_{avg}$" + (': {:.3f}%').format(emu_error_perc), fontsize=10)  # r'$\langle$(emu$_{mean}$ - model) / emu$\rangle$'

    xl = np.linspace(-4, 4, 100)
    plt.plot(xl, 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * xl**2), label='N(0,1)')
    plt.xlabel('z-score', fontsize=11)
    plt.xticks(np.arange(-8, 10, 2))
    plt.xlim([-8, 8])
    plt.ylim([0.0, 0.5])
    plt.legend()
    # Saving figure
    if save_fig: plt.savefig(fname)


def make_emulator(X, Y, par_limits, pcacomps, whiten=True, plot_expl_var=False, **kwargs):
    """
    Make an emulator.

    Parameters
    ----------
    X : array
        The parametisations which were used to generate the training data.
    Y : array
        The training data.
    par_limits : array
        The parameters' limits used.
    pcacomps : int
        How many first principal components are emulated.
    whiten : bool, optional
        Determines if the whitening is done in PCA. The default is True.
    plot_expl_var : bool, optional
        Determines if the explained variance by first npcs are to be plotted. This could be useful
        to know how well first npcs explain data. The default is False.

    Returns
    -------
    pcgp : class 'PCA_emulator.PCA_GPE'
        The emulator object from class PCA_emulator.PCA_GPE.

    """
    pcgp = pca_emu.PCA_GPE(pcacomps, whiten)
    pcgp.train(X, Y, par_limits, **kwargs)
    #  This is for seeing how much variance is explained by first npcs in a plot form.
    if plot_expl_var:
        plt.figure()
        plt.plot(np.arange(0, pcacomps), np.array(1 - pcgp.pca.explained_variance_ratio_)[:pcacomps])
        # plt.yscale("log")
        plt.title("Explained variance of pcs")
        plt.xlabel("pcs")
        plt.show()
    return pcgp


def start_sampling(par_limits, par_names, log_prob_emulator, saveMCMC=False, fname='MCMC_samples.dat', *args, **kwargs):
    """
    Start Monte Carlo Markov chain sampling.

    Parameters
    ----------
    par_limits : array
        The parameters' limits used.
    par_names : array
        Names of the parameters.
    log_prob_emulator : function
        The log-probability function which return log-probability given a parametisation.
    saveMCMC : bool, optional
        If the samples from MCMC is saved to a file. The default is False.
    fname : string, optional
        To which file samples are saved to. The default is 'MCMC_samples.dat'.
    *args : list
        arguments passed to MCMC.mcmc_sampling.
    **kwargs : list
        keyword arguments passed to MCMC.mcmc_sampling.

    Returns
    -------
    samples : array
        The samples from mcmc method.

    """
    print("Starting MCMC sampling")
    samples = None
    samples = MCMC.mcmc_sampling(par_limits, log_prob_emulator, *args, **kwargs)
    if saveMCMC:
        np.savetxt(fname, samples, header=' '.join(par_names), comments='# ',
                   delimiter=' ')
    return samples


def make_posterior(training_parameters, training_data,
                   par_limits, labels,
                   experimental_data, experimental_cov,
                   pcacomps, MCMC_walkers, MCMC_burn, MCMC_steps,
                   calc_zscore, testing_parameters, testing_data,
                   plot_save=False, emulator_std=False, emulator_cov=True,
                   save_samples=False, load_samples=False, samples_file="samples/samples.dat",
                   create_emulator=True, save_emulator=False, load_emulator=False, emu_file="emulators/emulator",
                   kwargs_for_pca={}, kwargs_for_MCMC={}):
    """ 
    Work in progress. General function to making a posterior and using this 
    setup easier.
    """
    # Should calculate the posterior.

    # kwargs_pca = {key: value for key, value in kwargs.items() if key.startswith('pca_')}

    def check_data(var):
        for i in range(len(var)):
            if isinstance(var[i], list) or np.ndim(var[i]) > 2:
                check_data(var[i])
            elif isinstance(var[i], str):
                var[i] = np.loadtxt(var[i])

    (training_parameters, training_data,
     testing_parameters, testing_data) = check_data(
        [training_parameters, training_data, testing_parameters, testing_data])

    if load_emulator:
        if isinstance(emu_file, list) or isinstance(emu_file, np.ndarray):
            file = [open(emu, "rb") for emu in emu_file]
            pca_gpe = [pickle.load(file) for file in file]
        else:
            file = open(emu_file, "rb")
            pca_gpe = pickle.load(file)
    elif create_emulator:
        pca_gpe = []
        emu_file = [emu_file]
        for t_par, t_data, emu_file in zip(training_parameters, training_data, emu_file):
            emu = make_emulator(t_par, t_data, par_limits, pcacomps, **kwargs_for_pca)
            pca_gpe.append(emu)

            if save_emulator:
                file = open(emu_file, "wb")
                pickle.dump(emu, file)
        print("Emulator done.")

    model_error = (emulator_std or emulator_cov)
    emulator = []
    chi2s = []
    for pca_gpe, experimental_data, experimental_cov in zip(
            pca_gpe, experimental_data, experimental_cov):
        experimental_std = np.sqrt(np.diag(experimental_cov))
        emulator.append(lambda theta: pca_gpe.predict(theta, emulator_std,
                                                      emulator_cov))
        chi2s.append(lambda theta: prob_calc.log_posterior(experimental_data,
                                                           emulator, theta,
                                                           experimental_cov,
                                                           experimental_std,
                                                           par_limits,
                                                           model_error))
        # If one want's to skip z-score.
        if calc_zscore:
            emu_samps, emu_std = pca_gpe.predict(testing_parameters, return_std=1)
            z_score(emu_samps, emu_std, true_model=testing_data)

    chi2 = lambda theta: np.sum([ch[theta] for ch in chi2s])

    samples = None
    if load_samples:
        if np.size(load_samples) > 1:
            samples = []
            for fname in load_samples:
                samples += [np.loadtxt(fname)]
        else: samples = np.loadtxt(load_samples)
    elif not load_samples and not create_emulator:
        print("No emulator. No sampling.")
    else:
        samples = start_sampling(par_limits, chi2, saveMCMC=save_samples,
                                 fname=save_samples, nwalkers=MCMC_walkers,
                                 nwalks=MCMC_steps, burn=MCMC_burn,
                                 walkers_par_labels=labels, **kwargs_for_MCMC)
    if samples is not None:
        plotting(samples, par_limits, labels, save=plot_save, fname=plot_save)
