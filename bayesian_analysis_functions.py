"""
Here are some methods to help with coding many setups.
There is also a make_posterior function for relatively easy creation of
Bayesian inference and the posterior.

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
             save=False, fname=None, fig=None, ax=None, zoom_width=3):
    """
    Plot marginal distributions and correlations of parameters by histograms from samples.

    Parameters
    ----------
    samples : array or list of arrays
        The chain of parameters from MCMC. Can be a list of several chains for plotting multiple
        posteriors to the same figure.
    par_limits : array, optional
        Limits of the parameters in the posterior. The default is None.
    labels : array, optional
        Names of each parameter. The default is None.
    zoom : string or array, optional
        Defines if distribution is zoomed to relevant region. The default is None, meaning whole
        parameter space is shown. Other options are 'auto' for zooming into clear posterior,
        or array of limits for parameters for custom zoom.
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
    zoom_width : int or float, optional
        If zoom="auto", this determines how wide the zoom will be.
        Might sometimes need adjusting manually.
        Recommended to use if the posterior is cut too early. The default is 3.

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
        par_limits = find_region(samples, zoom_width)
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
        # To get right titles, altough the quintiles can still be misleading
        # for some reason.
        # if par_limits is not None:
        #     samples = cut_samples(samples, par_limits)
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


def find_region(samples, width=3):
    """
    Find automatically the relevant region for the posterior distribution.

    Parameters
    ----------
    samples : array or list of arrays
        The chain of parameters from MCMC. Can be a list of several chains from several posteriors.
        The chain(s) must be flattened.
    width : int or float, optional
        Determines how wide the zoom will be. Might sometimes need adjusting manually.
        Recommended to use if the posterior is cut too early. The default is 3.

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
    for i in range(nparameters):
        par_limits += [[mean[i] - width * std[i], mean[i] + width * std[i]]]
    return np.array(par_limits)


def cut_samples(samples, par_limits):
    """
    Cut the samples given some parameter limitations.

    Parameters
    ----------
    samples : ndarray
        The parameter samples to be cut.
    par_limits : list or ndarray
        The limitations for parameters.

    Returns
    -------
    ndarray
        Cut down samples.

    """
    cut = []
    for i in range(par_limits.shape[0]):
        cut.append((par_limits[i, 0] < samples[:, i]) & (samples[:, i] < par_limits[i, 1]))
    cut = np.prod(np.array(cut), axis=0).astype(bool)
    return samples[cut, :]


def get_MAP(samples, chi2_func):
    """
    Calculate the maximum a priori (or mode) of each parameter round 3 decimals.

    Parameters
    ----------
    samples : ndarray
        The samples of the posterior.

    Returns
    -------
    Maximum a priori or MAP for each parameter.
    """
    return samples[np.argmin(np.abs([chi2_func(i) for i in samples]))]
    # return np.quantile(samples, 0.5, axis=0)


def z_score(pred_samps, pred_std, true_model, zoom=False, save_fig=False,
            fname='z_score.pdf', pred_mean=None, title=''):
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
        Name of the saved plot file, if save_fig is True. The default is 'z_score.pdf'.
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
    plt.figure(figsize=(5, 5))
    if zoom: rang = [-10, 10]
    else: rang = None
    plt.hist(z, 500, range=rang, density=True)  # Plot mean as a vertical line

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=0.9, alpha=0.8)
    plt.axvline(mean + std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    plt.axvline(mean - std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    if title != '': title = title + '\n'
    plt.title(title + ('Mean: {:.3f}, Std: {:.3f} \n').format(mean, std) + r"$\Delta_{avg}$" + (': {:.3f}%').format(emu_error_perc), fontsize=10)  # r'$\langle$(emu$_{mean}$ - model) / emu$\rangle$'

    xl = np.linspace(-4, 4, 100)
    plt.plot(xl, 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * xl**2), label=r'$\mathcal{N}(0,1)$')
    plt.xlabel('z-score', fontsize=11)
    plt.xticks(np.arange(-8, 10, 2))
    plt.xlim([-8, 8])
    plt.ylim([0.0, 0.5])
    plt.legend()
    # Saving figure
    if save_fig: plt.savefig(fname)
    # plt.savefig("z_score.pdf")


def make_cov(uncorr, beta):
    """
    Generate covariance matrix.

    Parameters
    ----------
    uncorr : nparray
        Uncorrelated uncertainties.
    beta : nparray
        Correlated uncertainties.

    Returns
    -------
    cov : nparray
        The covariance matrix

    """
    N = uncorr.size
    delta = np.eye(N)
    cov = uncorr**2 * delta + np.einsum('ik, jk', beta, beta)
    return cov


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
    samples = MCMC.mcmc_sampling(par_limits, log_prob_emulator,
                                 walkers_par_labels=par_names, *args, **kwargs)
    if saveMCMC:
        np.savetxt(fname, samples, header=' '.join(par_names), comments='# ',
                   delimiter=' ')
    return samples


def make_posterior(training_parameters,
                   training_data,
                   testing_parameters,
                   testing_data,
                   experimental_data,
                   experimental_cov,
                   parameters_limits,
                   parameters_labels,
                   MCMC_walkers=20,
                   MCMC_burn=50,
                   MCMC_steps=100,
                   n_principal_components=3,
                   calc_zscore=True,
                   only_zscore=False,
                   save_figs=False,
                   posterior_fname=None,
                   emulator_std=False,
                   emulator_cov=True,
                   save_samples=False,
                   load_samples=False,
                   MCMC_samples_file="data/MCMC/samples.dat",
                   create_emulator=True,
                   save_emulator=False,
                   load_emulator=False,
                   emulator_file="emulators/emulator",
                   kwargs_PCA_GPE={},
                   kwargs_MCMC={},
                   kwargs_plotting={}):
    """
    General function for easier use of this Bayesian inference and the
    posterior creation. Should make automatically the posterior. Can be given
    multiple datasets.

    Only needs training samples with corresponding training data for
    emulator(s), independent testing parameters and data, experimental data to
    fit to, and parameter limits and their names.

    One should tune MCMC settings for better posterior. Also possible to make
    tunings to PCA_GPE via passing kwargs to make_emulator function,
    to MCMC with kwargs_MCMC, and to plotting function with kwargs_plotting.
    For plotting I recommend to use "zoom": 'auto' kwarg for automatic zoom to
    posterior.

    Can save or load emulators to/from a file.

    TODO: Bug testing.
    Might still have some bugs, maybe at kwargs functionality.

    Parameters
    ----------
    training_parameters : ndarray or string
        Can be given the training parameter set as a ndarray or as string to
        the file containing the ndarray. Loads from the file with
        np.loadtxt(). Should correspond to training dataset(s).
    training_data : ndarray, list of ndarrays, string, list of strings
        The dataset(s) or path(s) to training samples/files. Has to correspond
        to training parameters. If given multiple, makes multiple emulators.
    testing_parameters : ndarray or string
        The dataset or path to testing parameter samples/file. Loads from the
        file with np.loadtxt(). Not needed if calc_zscore is False.
    testing_data : ndarray, list of ndarrays, string, list of strings
        The dataset(s) or path(s) to training samples/files. Has to correspond
        to testing_parameters. If given multiple, assumes and tests multiple
        emulators. Not needed if calc_zscore is False.
    experimental_data : ndarray, list of ndarrays, string, list of strings
        The experimental dataset(s) to fit the emulator to. Can have multiple,
        but must have same amount of datasets as there are experimental_covs
        and emulators.
    experimental_cov : ndarray, list of ndarrays
        The covariance matrix(es) for corresponding experimental dataset(s).
        Must have as many as there are experimental_data and emulators.
    parameters_limits : array
        The array of limits for parameters.
    parameters_labels : array
        The array made of names for parameters.
    MCMC_walkers : int, optional
        How many walkers are used in MCMC. Can be hunderds. The default is 20.
    MCMC_burn : int, optional
        How many first steps of walkers are discounted. The calibration stage.
        The default is 50.
    MCMC_steps : int, optional
        How many steps the walkers take after the burn stage.
        The default is 100.
    n_principal_components : int, optional
        How many principal components will be considered at GPE. Higher makes
        more accurate posterior, but slows down the process. 2 to 10 should be
        enough. Higher numbers also give warnings from scikit learn.
        The default is 3.
    calc_zscore : bool, optional
        If the z-score is calculated. The default is True.
    only_zscore : bool, optional
        If one wants to only calculate z-score distribution(s) for emulator(s).
        The default is False.
    save_figs : bool, optional
        TODO: Make to work. Now saves many figures as default.
        If the figures are stored as pdf. Work in progress.
        The default is False.
    posterior_fname : string, optional
        The file (with path) to which the posterior figure is saved.
        The default is None.
    emulator_std : bool, optional
        If the emulator(s) returns standard deviation as unccertainty estimate.
        The default is False.
    emulator_cov : bool, optional
        If the emulator(s) return covariance matrix as uncertainty estimate.
        The default is True.
    save_samples : bool or string, optional
        If the MCMC samples are saved to a file. Can be also given as a file
        where the samples will be stored then. The default is False.
    load_samples : bool, optional
        If the MCMC samples are loaded from a file. If given a string(s)
        path(s) to file(s), loads samples dataset(s). The default is False.
    MCMC_samples_file : TYPE, optional
        DESCRIPTION. The default is "data/MCMC/samples.dat".
    create_emulator : bool, optional
        If the emulator(s) is(are) to be created. The default is True.
    save_emulator : bool or string or list, optional
        Determines if the emulator(s) is/are saved to a file(s).
        If given a string address to a file or multiple files,
        saves one or multiple emulators to those files. The default is False.
    load_emulator : bool or string or list, optional
        Determines if the emulator(s) is/are loaded from a file(s).
        If given a string path(s) to a file or multiple files,
        loads one or multiple emulators. The default is False.
    emulator_file : string or list, optional
        Name(s) or file(s) (with path(s)) for the emulator(s) to load from or save to.
        The default is "emulators/emulator".
    kwargs_PCA_GPE : dict, optional
        The kwargs to pass on to pca method. The default is {}.
    kwargs_MCMC : dict, optional
        The kwargs to pass on to mcmc function. The default is {}.
    kwargs_plotting : dict, optional
        The kwargs to pass on to plotting stage, like {"zoom": "auto"}.
        The default is {}.

    Returns
    -------
    samples : ndarray
        The MCMC samples, from which the posterior figure is made.

    """

    # Reads data from txt file(s). One part can have multiple datasets.
    # Like no covariance and with covariance matrix datasets.
    def check_data(var):
        for i in range(len(var)):
            if isinstance(var[i], list) or np.ndim(var[i]) > 2:
                check_data(var[i])
            elif isinstance(var[i], str):
                var[i] = np.loadtxt(var[i])
        return var

    # Reads data from txt file(s). One part can have multiple datasets.
    # Like no covariance and with covariance matrix datasets.
    (training_parameters, training_data,
     testing_parameters, testing_data,
     experimental_data, experimental_cov) = check_data([training_parameters,
                                                        training_data,
                                                        testing_parameters,
                                                        testing_data,
                                                        experimental_data,
                                                        experimental_cov])

    def to_list_form(var):
        if not isinstance(var, list):
            return [var]
        return var

    training_data = to_list_form(training_data)
    testing_data = to_list_form(testing_data)
    experimental_data = to_list_form(experimental_data)
    experimental_cov = to_list_form(experimental_cov)

    pca_gpe = []
    if load_emulator:
        # If given multiple emulators
        if isinstance(load_emulator, str) or isinstance(load_emulator, list):
            emulator_file = load_emulator
        if isinstance(emulator_file, list) or isinstance(emulator_file, np.ndarray):
            file = [open(emu, "rb") for emu in emulator_file]
            pca_gpe = [pickle.load(file) for file in file]
        else:
            file = open(emulator_file, "rb")
            pca_gpe = [pickle.load(file)]
    elif create_emulator:
        if isinstance(save_emulator, str) or isinstance(save_emulator, list):
            emulator_file = save_emulator
        # TODO: If one wants to use different amounts of npcs.
        emulator_file = to_list_form(emulator_file)

        # If emulator file count is less than amount of training datasets.
        # To work with zip.
        i = 1
        while True:
            i += 1
            if len(emulator_file) != len(training_data):
                emulator_file.append(emulator_file[0] + "_" + str(i))
            else:
                break

        for tr_data, emu_file in zip(training_data, emulator_file):
            emu = make_emulator(training_parameters, tr_data, parameters_limits,
                                n_principal_components, **kwargs_PCA_GPE)
            pca_gpe.append(emu)

            if save_emulator:
                file = open(emu_file, "wb")
                pickle.dump(emu, file)

        print("Emulator done.")
    else:
        print("No emulator")

    model_error = (emulator_std or emulator_cov)
    if pca_gpe:
        if len(pca_gpe) < len(experimental_data):
            raise Exception("Less emulators than experimental datasets to predict")

        def chi2(theta):
            chi2s = []
            for pcagpe, expe_data, expe_cov in zip(
                    pca_gpe, experimental_data, experimental_cov):
                em = lambda theta: pcagpe.predict(theta, emulator_std, emulator_cov)
                chi2s.append(prob_calc.log_posterior(expe_data,
                                                     em,
                                                     theta,
                                                     parameters_limits,
                                                     expe_cov,
                                                     model_error=model_error))
            return np.sum(chi2s)

        if calc_zscore or only_zscore:  # If one want's to skip z-score.
            i = 0
            for test_data, pcagpe in zip(testing_data, pca_gpe):
                i += 1
                pred_samps = pcagpe.sample_y(testing_parameters, n_samples=100)
                pred_mean, pred_std = pcagpe.predict(testing_parameters,
                                                     return_std=True,
                                                     return_cov=False)
                z_score(pred_samps, pred_std, test_data, pred_mean=pred_mean)
            if only_zscore: return
    samples = None
    if load_samples:
        if not isinstance(load_samples, bool):
            if np.size(load_samples) > 1:
                samples = []
                for fname in load_samples:
                    samples += [np.loadtxt(fname)]
            else: samples = np.loadtxt(load_samples)
        else: samples = np.loadtxt('MCMC_samples.dat')
    elif not load_samples and not create_emulator:
        print("No emulator. No sampling. No samples to load.")
    elif pca_gpe:
        samples = start_sampling(parameters_limits, parameters_labels, chi2,
                                 saveMCMC=save_samples,
                                 fname=save_samples, nwalkers=MCMC_walkers,
                                 nwalks=MCMC_steps, burn=MCMC_burn,
                                 **kwargs_MCMC)
    if samples is not None:
        # If not flattened samples, flatten them for plotting.
        if samples.ndim > 2:
            sams = samples.reshape(-1, samples.shape[2])
        plotting(sams, parameters_limits, parameters_labels, save=save_figs,
                 fname=posterior_fname, **kwargs_plotting)
    return samples
