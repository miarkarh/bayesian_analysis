# -*- coding: utf-8 -*-
"""
Main file for light quark and charm setups with GBW dipole theory.

TODO: spikes from setups to same plot

@author: Mikko Artturi Karhunen
"""
import pickle
import numpy as np
from plyer import notification  # For getting notification, when the run is finished.
import time
import probability_formulas as prob_calc
import bayesian_analysis_functions as baf


def main_noC(saveMCMC=False, loadMCMC=False, fname=None,
             save_emulator=False, load_emulator=False,
             pcacomps=5, n_restarts=1, extra_std=0,
             nwalkers=100, nwalks=500, burn=200, flat=False,
             zoom=None, plot_save=False, plot_fname=None,
             zscore=False, z_zoom=False, z_save_fig=False, zfname='z_score.png', only_z=False,
             create_emulator=True, whiten=True,
             emu_std=False, emu_cov=False, cov=False,
             more_plots=True):
    """
    Make a bayesian analysis with light quark data.

    Parameters
    ----------
    saveMCMC : bool, optional
        If the MCMC samples are to be saved. The default is False
    loadMCMC : bool, optional
        If the MCMC samples are to be loaded from the "fname" file. The default is False.
    fname : string, optional
        The name of the file where the samples are loaded from. The default is None.
    save_emulator : bool, optional
        If the emulator is to be saved. The default is False.
    load_emulator : bool, optional
        If the emulator is loaded. The default is False.
    pcacomps : int, optional
        How many first npc are to be emulated. The default is 5.
    n_restarts : int, optional
        How many restarts the GPE takes for optimatization to training data. The default is 1.
    extra_std : float or array, optional
        This is added to emulator's error in pc-space, so it also adds to emulator's covariance.
        The default is 0.
    nwalkers : int, optional
        How many walkers are used in MCMC. The default is 100.
    nwalks : int, optional
        How many steps the walkers take in MCMC. The default is 500.
    burn : int, optional
        How many first steps are removed from sample data of MCMC. The default is 200.
    flat : bool, optional
        If the sample chains from MCMC are returned flattened to one chain, or for each walker their
        own chain. The default is False.
    zoom : string, optional
        If the posterior plots are to be zoomed to relevant regions. The default is None.
        Other options are 'auto'.
    plot_save : bool, optional
        If the posterior plot is to be saved. The default is False.
    plot_fname : string, optional
        The name of the file to which the posterior plot is saved. The default is None.
    zscore : bool, optional
        If the z-score is to be produced. The default is False.
    z_zoom : bool, optional
        If the z-score plot is to be zoomed. The default is False.
    z_save_fig : bool, optional
        If the z-score plot is to be saved. The default is False.
    zfname : string, optional
        The filename to which z-splot is saved at. The default is 'z_score.png'.
    only_z : bool, optional
        If only the z-score is produced. The default is False.
    create_emulator : bool, optional
        If the emulator is created. The default is True.
    whiten : bool, optional
        If the whitening is done in pca. The default is True.
    emu_std : bool, optional
        If the emulators standard deviation is taken as it's error. The default is False.
    emu_cov : bool, optional
        If the emulators standard deviation is taken as it's error. Overwrites the emu_std.
        The default is False.
    cov : bool, optional
        If the experimental covariance is taken into calculation. Otherwise the experimental
        standard deviations are taken. The default is False.
    more_plots : bool, optional
        If one wants to get some more plots like fit to data. The default is False.

    Returns
    -------
    None.

    """
    par_limits = baf.parameter_limits(3)
    labels = baf.parameter_names(3)
    light_data = baf.load_light_data()

    # experimental data
    Q2, x, y, sigma_r_exp = light_data[:, 0:4].T
    stat, unc = light_data[:, 4:6].T * 0.01 * sigma_r_exp
    procedural = (light_data[:, -7:].T * 0.01 * sigma_r_exp).T
    uncorr = np.sqrt(stat**2 + unc**2 + np.sum(procedural**2, axis=1))
    tot_noproc = light_data[:, -8] * 0.01 * sigma_r_exp

    ps1 = np.loadtxt('data/training data/parameters/lhc_samples_200_noC.dat')
    yt1 = np.loadtxt('data/training data/results/sigma_r_training_200_noC.dat')
    parameter_samples = ps1
    y_training = yt1

    beta = (light_data[:, 6:-9].T * 0.01 * sigma_r_exp).T
    beta = np.column_stack((beta, procedural))
    if cov:
        cov = baf.make_cov(uncorr, beta)
        # This is for that only covariance matrix is used later
        sigma_r_err = None

    # If one does not want to calculate with covariance matrix
    else:
        sigma_r_err = np.sqrt(tot_noproc**2 + np.sum(procedural**2, axis=1))
        print("Mean of experimental error (percent):")
        print(np.mean(sigma_r_err / sigma_r_exp * 100))
        # This is for that covariance is not used later
        cov = None

    if create_emulator:
        if not load_emulator:
            pca_gpe = baf.make_emulator(parameter_samples, y_training, par_limits, pcacomps, whiten, n_restarts=1,
                                        consta_k1_bounds=[1e-5, 1e5], length_scale_bounds=[0.001, 1000], noise_level_bounds=[1e-19, 1e1])
            if save_emulator:
                file = open("emulators/emulator_noC", "wb")
                pickle.dump(pca_gpe, file)
        else:
            file = open("emulators/emulator_noC", "rb")
            pca_gpe = pickle.load(file)

        # extra_std = 0.03#.21*np.mean(tot_noproc)
        emulator = lambda theta: pca_gpe.predict(theta, return_std=emu_std, return_cov=emu_cov, extra_std=extra_std)
        log_prob_emulator = lambda theta: prob_calc.log_posterior(sigma_r_exp, emulator, theta,
                                                                  cov_y=cov, ystd=sigma_r_err, par_limits=par_limits,
                                                                  model_error=(emu_std or emu_cov))
        print("emulator ready")

        # If one want's to skip z-score.
        if zscore:
            sample_size = 200
            model = np.loadtxt("data/testing data/sigma_r_testing_full_" + str(sample_size) + '.dat')

            test_samples = np.loadtxt("data/testing data/test_parameters_" + str(sample_size) + '_samples_' + str(par_limits.shape[0]) + '_pars.dat')
            pred = pca_gpe.sample_y(test_samples, n_samples=100)
            pred_std = pca_gpe.predict(test_samples, return_std=True, return_cov=False, extra_std=extra_std)[1]

            zfname = "z_score_noC"
            if cov is not None:
                zfname = zfname + "_with_cov"
            zfname = zfname + ".png"

            baf.z_score(pred, pred_std, model, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)
            if only_z: return

    samples = None
    if loadMCMC:
        samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator. No sampling.")
    else:
        samples = baf.start_sampling(par_limits, log_prob_emulator, saveMCMC, fname, nwalkers, nwalks, burn, walkers_par_labels=labels, flat=flat)
    # Plots posterior and takes plotted axis limits.
    if samples is not None:
        fig, post_limits, ax = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)

    # Next is plotting against experimental data.
    if more_plots:
        # theta = [14.670, 0.306, 2.044] #nocov
        # the = [13.383, 0.318, 2.333] #wcov

        D = sigma_r_exp
        T = np.loadtxt("data/testing data/sigma_r_some_thetas_2.dat")[0]
        # Tcov = np.loadtxt("data/testing data/sigma_r_some_thetas_2.dat")[1]
        Tcov = np.loadtxt("data/testing data/sigma_r_cov_par_noC.dat")  # theta = 13.711 0.312 2.217
        baf.more_plots(D, T, Tcov, beta, uncorr, Q2, x)

        model = lambda x: Tcov
        # This does not matter. Just so code works.
        theta = [0, 0, 0]
        chi2 = prob_calc.log_likelihood(D, model, theta, cov_y=cov, model_error=False)
        print("chi2/N nocov:")
        print(chi2 / D.shape[0])

        # sigma_r_err = np.sqrt(tot_noproc**2 + np.sum(procedural**2, axis=1))
        # sigma_r_err = np.sqrt(np.sum(beta**2, axis=1) + uncorr**2)
        # hundred_samples(samples, post_limits, emulator, x, sigma_r_exp, sigma_r_err)


def main_C(saveMCMC=False, loadMCMC=False, fname=None,
           save_emulator=False, load_emulator=False,
           pcacomps=5, n_restarts=1, extra_std=0,
           nwalkers=100, nwalks=500, burn=200, flat=False,
           zoom=None, plot_save=False, plot_fname=None,
           zscore=False, z_zoom=False, z_save_fig=False, zfname='z_score.png', only_z=False,
           create_emulator=True, whiten=True,
           emu_std=False, emu_cov=False, cov=False,
           more_plots=False):
    """
    Make a bayesian analysis with light quark data.

    Parameters
    ----------
    saveMCMC : bool, optional
        If the MCMC samples are to be saved. The default is False
    loadMCMC : bool, optional
        If the MCMC samples are to be loaded from the "fname" file. The default is False.
    fname : string, optional
        The name of the file where the samples are loaded from. The default is None.
    save_emulator : bool, optional
        If the emulator is to be saved. The default is False.
    load_emulator : bool, optional
        If the emulator is loaded. The default is False.
    pcacomps : int, optional
        How many first npc are to be emulated. The default is 5.
    n_restarts : int, optional
        How many restarts the GPE takes for optimatization to training data. The default is 1.
    extra_std : float or array, optional
        This is added to emulators' error in pc-space, so it also adds to emulator's covariance.
        The default is 0. Can be in shape of (extra_std_li, extra_std_c) for 2 emulators.
    nwalkers : int, optional
        How many walkers are used in MCMC. The default is 100.
    nwalks : int, optional
        How many steps the walkers take in MCMC. The default is 500.
    burn : int, optional
        How many first steps are removed from sample data of MCMC. The default is 200.
    flat : bool, optional
        If the sample chains from MCMC are returned flattened to one chain, or for each walker their
        own chain. The default is False.
    zoom : string, optional
        If the posterior plots are to be zoomed to relevant regions. The default is None.
        Other options are 'auto'.
    plot_save : bool, optional
        If the posterior plot is to be saved. The default is False.
    plot_fname : string, optional
        The name of the file to which the posterior plot is saved. The default is None.
    zscore : bool, optional
        If the z-score is to be produced. The default is False.
    z_zoom : bool, optional
        If the z-score plot is to be zoomed. The default is False.
    z_save_fig : bool, optional
        If the z-score plot is to be saved. The default is False.
    zfname : string, optional
        The filename to which z-splot is saved at. The default is 'z_score.png'.
    only_z : bool, optional
        If only the z-score is produced. The default is False.
    create_emulator : bool, optional
        If the emulator is created. The default is True.
    whiten : bool, optional
        If the whitening is done in pca. The default is True.
    emu_std : bool, optional
        If the emulators standard deviation is taken as it's error. The default is False.
    emu_cov : bool, optional
        If the emulators standard deviation is taken as it's error. Overwrites the emu_std.
        The default is False.
    cov : bool, optional
        If the experimental covariance is taken into calculation. Otherwise the experimental
        standard deviations are taken. The default is False.
    more_plots : bool, optional
        If one wants to get some more plots like fit to data. The default is False.

    Returns
    -------
    None.

    """
    par_limits = baf.parameter_limits(4)
    labels = baf.parameter_names(4)

    ps1 = np.loadtxt('data/training data/parameters/lhc_samples_300_wC.dat')

    ytli1 = np.loadtxt('data/training data/results/sigma_r_training_300_light_c.dat')

    # only c quark
    ytc1 = np.loadtxt('data/training data/results/sigma_r_training_300_c.dat')

    parameter_samples = ps1
    y_trainingli = ytli1
    y_trainingc = ytc1

    light_data = baf.load_light_data()

    # exp for experimental data
    Q2li, xli, yli, sigma_r_expli = light_data[:, 0:4].T
    tot_no_proc = light_data[:, -8] * 0.01 * sigma_r_expli  # tot_no_proc
    procedural = (light_data[:, -7:].T * 0.01 * sigma_r_expli).T
    sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))

    # c quark
    cdata = baf.load_c_data()

    Q2c, xc, sigma_r_expc = cdata[:, 0:3].T
    # yc = Q2c / (318**2 * xc)
    statc, uncc = cdata[:, (3, 4)].T * 0.01 * sigma_r_expc
    uncorrc = np.sqrt(statc**2 + uncc**2)
    betac = (cdata[:, 5:].T * 0.01 * sigma_r_expc).T
    sys_erc = np.sqrt(np.sum(betac**2, axis=1))
    sigma_r_errc = np.sqrt(uncorrc**2 + sys_erc**2)
    print("mean of experimental error (percent):")
    print(np.mean(np.abs(sigma_r_errli / sigma_r_expli * 100)))
    print(np.mean(np.abs(sigma_r_errc / sigma_r_expc * 100)))

    betali = (light_data[:, 6:-9].T * 0.01 * sigma_r_expli).T
    statli, uncli = light_data[:, 4:6].T * 0.01 * sigma_r_expli
    uncorrli = np.sqrt(statli**2 + uncli**2 + np.sum(procedural**2, axis=1))
    if cov:
        cov_li = baf.make_cov(uncorrli, betali)
        cov_c = baf.make_cov(uncorrc, betac)
        # This is for that only covariance matrix is used later
        sigma_r_errli = None
        sigma_r_errc = None
    else:
        cov_li = None
        cov_c = None

    if create_emulator:
        if not load_emulator:
            pca_gpeli = baf.make_emulator(parameter_samples, y_trainingli, par_limits, pcacomps, whiten, n_restarts, noise_level_bounds=[1e-10, 1e-5])
            pca_gpec = baf.make_emulator(parameter_samples, y_trainingc, par_limits, pcacomps, whiten, n_restarts, length_scale_bounds=[0.001, 1000], noise_level_bounds=[1e-13, 1e-1])
            if save_emulator:
                file = open("emulators/emulator_wCli", "wb")
                pickle.dump(pca_gpeli, file)
                file = open("emulators/emulator_wCc", "wb")
                pickle.dump(pca_gpec, file)
        else:
            file = open("emulators/emulator_wCli", "rb")
            pca_gpeli = pickle.load(file)
            file = open("emulators/emulator_wCc", "rb")
            pca_gpec = pickle.load(file)
        print("Training done.")

        if np.ndim(extra_std) == 2:
            extra_stdli = extra_std[0]
            extra_stdc = extra_std[1]
        else:
            extra_stdli = extra_std
            extra_stdc = extra_std
        emulatorli = lambda theta: pca_gpeli.predict(theta, return_std=emu_std, return_cov=emu_cov, extra_std=extra_stdli)
        emulatorc = lambda theta: pca_gpec.predict(theta, return_std=emu_std, return_cov=emu_cov, extra_std=extra_stdc)

        chi2_li = lambda theta: prob_calc.log_posterior(sigma_r_expli, emulatorli, theta,
                                                        cov_y=cov_li, ystd=sigma_r_errli, par_limits=par_limits,
                                                        model_error=(emu_std or emu_cov))
        chi2_c = lambda theta: prob_calc.log_posterior(sigma_r_expc, emulatorc, theta,
                                                       cov_y=cov_c, ystd=sigma_r_errc, par_limits=par_limits,
                                                       model_error=(emu_std or emu_cov))
        log_prob_emulator = lambda theta: chi2_li(theta) + chi2_c(theta)

        # If one want's to skip z-score.
        if zscore:
            sample_size = 200
            test_samples = np.loadtxt("data/testing data/test_parameters_" + str(sample_size) + '_samples_' + str(par_limits.shape[0]) + '_pars.dat')
            modelli = np.loadtxt("data/testing data/sigma_r_testing_" + str(sample_size) + '_light_c.dat')
            modelc = np.loadtxt("data/testing data/sigma_r_testing_" + str(sample_size) + '_c.dat')
            model = np.concatenate((modelli, modelc), axis=1)

            sample_y_li = lambda theta: pca_gpeli.sample_y(theta, n_samples=100)
            sample_y_c = lambda theta: pca_gpec.sample_y(theta, n_samples=100)
            emul_std_li = lambda theta: pca_gpeli.predict(theta, return_std=True, return_cov=False, extra_std=extra_stdli)[1]
            emul_std_c = lambda theta: pca_gpec.predict(theta, return_std=True, return_cov=False, extra_std=extra_stdc)[1]
            emulator_draw = lambda theta: np.concatenate((sample_y_li(theta), sample_y_c(theta)), axis=2)
            emulator_std = lambda theta: np.concatenate((emul_std_li(theta), emul_std_c(theta)), axis=1)
            zfname = "z_score_wC"
            if cov is not None:
                zfname = zfname + "_with_cov"
            zfname = zfname + ".png"

            print("Total emulator")
            pred = emulator_draw(test_samples)
            pred_std = emulator_std(test_samples)
            baf.z_score(pred, pred_std, model, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)

            print("Light data emulator")
            pred = sample_y_li(test_samples)
            pred_std = emul_std_li(test_samples)
            baf.z_score(pred, pred_std, modelli, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)

            print("Charm data emulator")
            pred = sample_y_c(test_samples)
            pred_std = emul_std_c(test_samples)
            baf.z_score(pred, pred_std, modelc, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)

            if only_z: return

    samples = None
    if loadMCMC:
        samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator. No sampling.")
    else:
        # import emcee.moves
        # moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        moves = None
        samples = baf.start_sampling(par_limits, log_prob_emulator, saveMCMC, fname, nwalkers,
                                     nwalks, burn, walkers_par_labels=labels, moves=moves, flat=flat)

    # Plots posterior and takes plotted axis limits.
    if samples is not None:
        fig, post_limits, ax = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)

    if more_plots:
        # theta = 20.931 0.278 0.962 1.729 #nocov
        # theta with cov = 18.112 0.285 1.144 2.000 #cov

        Dli = sigma_r_expli
        Dc = sigma_r_expc

        Tc = np.loadtxt("data/testing data/sigma_r_c_wC.dat")
        Tli = np.loadtxt("data/testing data/sigma_r_c_light_wC.dat")

        Tcovc = np.loadtxt("data/testing data/sigma_r_c_wCwcov.dat")
        Tcovli = np.loadtxt("data/testing data/sigma_r_c_light_wCwcov.dat")

        baf.more_plots(Dli, Tli, Tcovli, betali, uncorrli, Q2li, xli)
        baf.more_plots(Dc, Tc, Tcovc, betac, uncorrc, Q2c, xc)

        modli = lambda t: Tli
        modc = lambda t: Tc

        cov_li = baf.make_cov(uncorrli, betali)
        cov_c = baf.make_cov(uncorrc, betac)

        chi2li = prob_calc.log_likelihood(Dli, modli, 1, ystd=sigma_r_errli, model_error=False) / 430
        chi2licov = prob_calc.log_likelihood(Dli, modli, 1, cov_y=cov_li, model_error=False) / 430

        chi2c = prob_calc.log_likelihood(Dc, modc, 1, ystd=sigma_r_errc, model_error=False) / 34
        chi2ccov = prob_calc.log_likelihood(Dc, modc, 1, cov_y=cov_c, model_error=False) / 34

        # This needs better design on looks. A bit confusing now.
        print("chi^2/N kokonaisvaikutusdatalle:")
        print(chi2li, chi2licov)
        print("chi^2/N vain charm datalle:")
        print(chi2c, chi2ccov)


def main_C_gamma(saveMCMC=False, loadMCMC=False, fname=None,
                 save_emulator=False, load_emulator=False,
                 pcacomps=5, n_restarts=1, extra_std=0,
                 nwalkers=100, nwalks=500, burn=200, flat=False,
                 zoom=None, plot_save=False, plot_fname=None,
                 zscore=False, z_zoom=False, z_save_fig=False, zfname='z_score.png', only_z=False,
                 create_emulator=True, whiten=True,
                 emu_std=False, emu_cov=False, cov=False,
                 more_plots=False):
    """
    Make a bayesian analysis with light quark data.

    Parameters
    ----------
    saveMCMC : bool, optional
        If the MCMC samples are to be saved. The default is False
    loadMCMC : bool, optional
        If the MCMC samples are to be loaded from the "fname" file. The default is False.
    fname : string, optional
        The name of the file where the samples are loaded from. The default is None.
    save_emulator : bool, optional
        If the emulator is to be saved. The default is False.
    load_emulator : bool, optional
        If the emulator is loaded. The default is False.
    pcacomps : int, optional
        How many first npc are to be emulated. The default is 5.
    n_restarts : int, optional
        How many restarts the GPE takes for optimatization to training data. The default is 1.
    extra_std : float or array, optional
        This is added to emulator's error in pc-space, so it also adds to emulator's covariance.
        The default is 0. Can be in shape of (extra_std_li, extra_std_c) for 2 emulators.
    nwalkers : int, optional
        How many walkers are used in MCMC. The default is 100.
    nwalks : int, optional
        How many steps the walkers take in MCMC. The default is 500.
    burn : int, optional
        How many first steps are removed from sample data of MCMC. The default is 200.
    flat : bool, optional
        If the sample chains from MCMC are returned flattened to one chain, or for each walker their
        own chain. The default is False.
    zoom : string, optional
        If the posterior plots are to be zoomed to relevant regions. The default is None.
        Other options are 'auto'.
    plot_save : bool, optional
        If the posterior plot is to be saved. The default is False.
    plot_fname : string, optional
        The name of the file to which the posterior plot is saved. The default is None.
    zscore : bool, optional
        If the z-score is to be produced. The default is False.
    z_zoom : bool, optional
        If the z-score plot is to be zoomed. The default is False.
    z_save_fig : bool, optional
        If the z-score plot is to be saved. The default is False.
    zfname : string, optional
        The filename to which z-splot is saved at. The default is 'z_score.png'.
    only_z : bool, optional
        If only the z-score is produced. The default is False.
    create_emulator : bool, optional
        If the emulator is created. The default is True.
    whiten : bool, optional
        If the whitening is done in pca. The default is True.
    emu_std : bool, optional
        If the emulators standard deviation is taken as it's error. The default is False.
    emu_cov : bool, optional
        If the emulators standard deviation is taken as it's error. Overwrites the emu_std.
        The default is False.
    cov : bool, optional
        If the experimental covariance is taken into calculation. Otherwise the experimental
        standard deviations are taken. The default is False.
    more_plots : bool, optional
        If one wants to get some more plots like fit to data. The default is False.

    Returns
    -------
    None.

    """
    par_limits = baf.parameter_limits(5)
    labels = baf.parameter_names(5)

    light_data = baf.load_light_data()

    # exp for experimental data
    Q2li, xli, yli, sigma_r_expli = light_data[:, 0:4].T
    tot_no_proc = light_data[:, -8] * 0.01 * sigma_r_expli  # tot_no_proc
    procedural = (light_data[:, -7:].T * 0.01 * sigma_r_expli).T
    # procedural = np.zeros_like(procedural)
    sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))

    # c quark
    cdata = baf.load_c_data()

    Q2c, xc, sigma_r_expc = cdata[:, 0:3].T
    # yc = Q2c / (318**2 * xc)
    statc, uncc = cdata[:, (3, 4)].T * 0.01 * sigma_r_expc
    uncorrc = np.sqrt(statc**2 + uncc**2)
    betac = (cdata[:, 5:].T * 0.01 * sigma_r_expc).T
    sys_erc = np.sqrt(np.sum(betac**2, axis=1))
    sigma_r_errc = np.sqrt(uncorrc**2 + sys_erc**2)
    print("mean of experimental error (percent):")
    print(np.mean(np.abs(sigma_r_errli / sigma_r_expli * 100)))
    print(np.mean(np.abs(sigma_r_errc / sigma_r_expc * 100)))

    ps1 = np.loadtxt('data/training data/parameters/lhc_samples_200_5par.dat')  # [::4]

    ytli1 = np.loadtxt('data/training data/results/sigma_r_training_200_c_light_5par.dat')

    ytc1 = np.loadtxt('data/training data/results/sigma_r_training_200_c_5par.dat')

    parameter_samples = ps1  # np.row_stack((ps1, ps2, ps289))
    y_trainingli = ytli1  # np.row_stack((ytli1, ytli2, ytli289))
    y_trainingc = ytc1  # np.row_stack((ytc1, ytc2, ytc289))

    betali = (light_data[:, 6:-9].T * 0.01 * sigma_r_expli).T
    statli, uncli = light_data[:, 4:6].T * 0.01 * sigma_r_expli
    uncorrli = np.sqrt(statli**2 + uncli**2 + np.sum(procedural**2, axis=1))
    if cov:
        cov_li = baf.make_cov(uncorrli, betali)
        cov_c = baf.make_cov(uncorrc, betac)
        # This is for that only covariance matrix is used later
        sigma_r_errli = None
        sigma_r_errc = None
    else:
        cov_li = None
        cov_c = None

    if create_emulator:
        if not load_emulator:
            pca_gpeli = baf.make_emulator(parameter_samples, y_trainingli, par_limits, pcacomps, whiten, n_restarts, length_scale_bounds=[1e-5, 1e5], noise_level_bounds=[1e-13, 1e-1])
            pca_gpec = baf.make_emulator(parameter_samples, y_trainingc, par_limits, pcacomps, whiten, n_restarts, noise_level_bounds=[1e-13, 1e-1])
            if save_emulator:
                file = open("emulators/emulator_wC_gamma_li", "wb")
                pickle.dump(pca_gpeli, file)
                file = open("emulators/emulator_wC_gamma_c", "wb")
                pickle.dump(pca_gpec, file)
        else:
            file = open("emulators/emulator_wC_gamma_li", "rb")
            pca_gpeli = pickle.load(file)
            file = open("emulators/emulator_wC_gamma_c", "rb")
            pca_gpec = pickle.load(file)
        print("Training done.")

        if np.ndim(extra_std) == 2:
            extra_stdli = extra_std[0]
            extra_stdc = extra_std[1]
        else:
            extra_stdli = extra_std
            extra_stdc = extra_std
        emulatorli = lambda theta: pca_gpeli.predict(theta, return_std=emu_std, return_cov=emu_cov, extra_std=extra_stdli)
        emulatorc = lambda theta: pca_gpec.predict(theta, return_std=emu_std, return_cov=emu_cov, extra_std=extra_stdc)

        chi2_li = lambda theta: prob_calc.log_posterior(sigma_r_expli, emulatorli, theta,
                                                        cov_y=cov_li, ystd=sigma_r_errli, par_limits=par_limits,
                                                        model_error=(emu_std or emu_cov))
        chi2_c = lambda theta: prob_calc.log_posterior(sigma_r_expc, emulatorc, theta,
                                                       cov_y=cov_c, ystd=sigma_r_errc, par_limits=par_limits,
                                                       model_error=(emu_std or emu_cov))
        log_prob_emulator = lambda theta: chi2_li(theta) + chi2_c(theta)

        # If one want's to skip z-score.
        if zscore:
            sample_size = 100
            test_samples = np.loadtxt("data/testing data/test_parameters_" + str(sample_size) + '_samples_' + str(par_limits.shape[0]) + '_pars.dat')
            modelli = np.loadtxt("data/testing data/sigma_r_testing_" + str(sample_size) + '_c_light_' + str(par_limits.shape[0]) + 'pars.dat')
            modelc = np.loadtxt("data/testing data/sigma_r_testing_" + str(sample_size) + '_c_' + str(par_limits.shape[0]) + 'pars.dat')
            model = np.concatenate((modelli, modelc), axis=1)

            sample_y_li = lambda theta: pca_gpeli.sample_y(theta, n_samples=100)
            sample_y_c = lambda theta: pca_gpec.sample_y(theta, n_samples=100)
            emul_std_li = lambda theta: pca_gpeli.predict(theta, return_std=True, return_cov=False, extra_std=extra_stdli)[1]
            emul_std_c = lambda theta: pca_gpec.predict(theta, return_std=True, return_cov=False, extra_std=extra_stdc)[1]
            emulator_draw = lambda theta: np.concatenate((sample_y_li(theta), sample_y_c(theta)), axis=2)
            emulator_std = lambda theta: np.concatenate((emul_std_li(theta), emul_std_c(theta)), axis=1)
            zfname = "z_score_wC"
            if cov is not None:
                zfname = zfname + "_with_cov"
            zfname = zfname + ".png"

            print("Total emulator")
            pred = emulator_draw(test_samples)
            pred_std = emulator_std(test_samples)
            baf.z_score(pred, pred_std, model, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)

            print("Light data emulator")
            pred = sample_y_li(test_samples)
            pred_std = emul_std_li(test_samples)
            baf.z_score(pred, pred_std, modelli, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)

            print("Charm data emulator")
            pred = sample_y_c(test_samples)
            pred_std = emul_std_c(test_samples)
            baf.z_score(pred, pred_std, modelc, zoom=z_zoom, save_fig=z_save_fig, fname=zfname)
        if only_z: return

    samples = None
    if loadMCMC:
        samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator. No sampling.")
    else:
        # import emcee.moves
        # moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        moves = None
        samples = baf.start_sampling(par_limits, log_prob_emulator, saveMCMC, fname, nwalkers,
                                     nwalks, burn, walkers_par_labels=labels, moves=moves, flat=flat)
    # Plots posterior and takes plotted axis limits.
    if samples is not None:
        fig, post_limits, ax = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)


if __name__ == '__main__':
    main_noC(1, 0, fname='data/MCMC/MCMC_noC.dat',
             save_emulator=0, load_emulator=1,
             pcacomps=10, n_restarts=10, extra_std=[0.00035],
             nwalkers=200, nwalks=1000, burn=500, flat=True,
             zoom='auto',  # plot_fname='kuvat/noC_temp.png',
             zscore=1, only_z=0,
             create_emulator=1, emu_std=0, emu_cov=1, cov=0,
             more_plots=False)

    # main_C(1, 0, fname='data/MCMC/MCMC_wC.dat',
    #        save_emulator=0, load_emulator=1,
    #        pcacomps=10, n_restarts=10, extra_std=[[0.00025], [0.00035]],
    #        nwalkers=200, nwalks=1000, burn=500, flat=True,
    #        zoom='auto',
    #        zscore=1, only_z=0,
    #        create_emulator=1, emu_std=0, emu_cov=1, cov=0,
    #        more_plots=False)

    # main_C_gamma(1, 0, fname='data/MCMC/MCMC_wCgamma.dat',
    #              save_emulator=0, load_emulator=1,
    #              pcacomps=10, n_restarts=10, extra_std=[[0.0055], [0.0075]],
    #              nwalkers=200, nwalks=1000, burn=500, flat=True,
    #              zoom='auto',  # plot_fname='kuvat/wC_temp.png',
    #              zscore=1, only_z=0,
    #              create_emulator=1, emu_std=0, emu_cov=1, cov=0,
    #              more_plots=False)

    # For getting notification when done. Needs plyer module.
    while True:
        time.sleep(3)
        notification.notify(
            title="Finished executing",
            message="Successful",
        )
        time.sleep(1)
        break
