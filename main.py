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
import setup_functions as sef


def main_noC(saveMCMC=False, loadMCMC=False, fname=None,
             save_emulator=False, load_emulator=False,
             pcacomps=5, n_restarts=1, extra_std=0,
             nwalkers=100, nwalks=500, burn=200, flat=False,
             zoom=None, plot_save=False, plot_fname=None,
             zscore=False, z_zoom=False, only_z=False,
             z_save_fig=False, zfname='z_score.png',
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
    fname : string or list of strings, optional
        The name (names) of the file (files) where the samples are loaded from. The default is None.
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
    par_limits = sef.parameter_limits(3)
    labels = sef.parameter_names(3)
    light_data = sef.load_light_data()

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
    # beta = np.column_stack((beta, procedural))
    if cov:
        cov = sef.make_cov(uncorr, beta)
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
            pred_samps = pca_gpe.sample_y(test_samples, n_samples=100)
            pred_mean, pred_std = pca_gpe.predict(test_samples, return_std=True, return_cov=False, extra_std=extra_std)

            zfname = "z_score_noC"
            if cov is not None:
                zfname = zfname + "_with_cov"
            zfname = zfname + ".pdf"
            ztitle = "GBW"
            baf.z_score(pred_samps, pred_std, model, zoom=z_zoom,
                        save_fig=z_save_fig, fname=zfname, pred_mean=pred_mean, title=ztitle)
            if only_z: return

    samples = None
    if loadMCMC:
        if np.size(fname) > 1:
            samples = []
            for fname in fname:
                samples += [np.loadtxt(fname)]
        else: samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator. No sampling.")
    else:
        samples = baf.start_sampling(par_limits, labels, log_prob_emulator, saveMCMC, fname, nwalkers, nwalks, burn, walkers_par_labels=labels, flat=flat)
    # Plots posterior and takes plotted axis limits.
    if samples is not None:
        fig, post_limits = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)

    # cut = []
    # for i in range(post_limits.shape[0]):
    #     cut += [(post_limits[i, 0] < samples[:, i]) & (samples[:, i] < post_limits[i, 1])]
    # cut = np.prod(np.array(cut), axis=0)
    # hundsamples = samples[cut.astype(bool), :]
    # hundsamples = hundsamples[np.random.randint(0, hundsamples.shape[0], 100)]
    # np.savetxt("light_posterior_hundred_samples.dat", hundsamples)
    # hundsamples = np.loadtxt("light_posterior_hundred_samples.dat")

    # Next is plotting against experimental data.
    if more_plots:
        # theta = [14.670, 0.306, 2.044] #nocov
        # the = [13.383, 0.318, 2.333] #wcov
        # if cov is not None:
        sampless = samples
        if isinstance(samples, list):
            post_limits = baf.find_region(samples[0])
            sampless[0] = sef.cut_samples(sampless[0], post_limits)
            post_limits = baf.find_region(samples[1])
            sampless[1] = sef.cut_samples(sampless[1], post_limits)
        else:
            post_limits = baf.find_region(samples)
            sampless = sef.cut_samples(sampless, post_limits)
        if create_emulator:
            # if isinstance(samples, list): sampless = samples[1]
            sigma_r_err = np.sqrt(tot_noproc**2 + np.sum(procedural**2, axis=1))
            sef.samples_plot(sampless[1], 5000, post_limits, emulator, x, Q2, y, sigma_r_exp, sigma_r_err)

        Ndof = sigma_r_exp.shape[0] - 3

        emula = lambda theta: emulator(theta)[0]
        chi2_nocov = lambda theta: prob_calc.log_likelihood(sigma_r_exp, emula, theta, ystd=sigma_r_err, model_error=False)
        theta_map = [14.96532493, 0.31417557, 2.07724475]  # baf.get_MAP(sampless[0], chi2_nocov)
        print("MAP for nocov: ", theta_map)
        print("chi2/N (MAP): ", chi2_nocov(theta_map) / Ndof)

        cov = sef.make_cov(uncorr, beta)
        chi2_cov = lambda theta: prob_calc.log_likelihood(sigma_r_exp, emula, theta, cov_y=cov, model_error=False)
        theta_map_cov = [13.53738093, 0.31737713, 2.30284369]  # baf.get_MAP(sampless[1], chi2_cov)
        print("MAP for cov: ", theta_map_cov)
        print("chi2/N wCov (MAP): ", chi2_cov(theta_map_cov) / Ndof)

        print("MAP cov chi no cov: ", theta_map_cov)
        print("chi2/Ndof (MAP): ", chi2_nocov(theta_map_cov) / Ndof)

        print("MAP chi wcov: ", theta_map)
        print("chi2/Ndof wCov (MAP): ", chi2_cov(theta_map) / Ndof)

        T = emulator(theta_map)[0]
        Tcov = emulator(theta_map_cov)[0]

        sef.more_plots(sigma_r_exp, T, Tcov, beta, uncorr, Q2, x)


def main_C(saveMCMC=False, loadMCMC=False, fname=None,
           save_emulator=False, load_emulator=False,
           pcacomps=5, n_restarts=1, extra_std=0,
           nwalkers=100, nwalks=500, burn=200, flat=False,
           zoom=None, plot_save=False, plot_fname=None,
           zscore=False, z_zoom=False, only_z=False,
           z_save_fig=False, zfname='z_score.png',
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
    fname : string or list of strings, optional
        The name (names) of the file (files) where the samples are loaded from. The default is None.
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
    par_limits = sef.parameter_limits(4)
    labels = sef.parameter_names(4)

    ps1 = np.loadtxt('data/training data/parameters/lhc_samples_300_wC.dat')

    ytli1 = np.loadtxt('data/training data/results/sigma_r_training_300_light_c.dat')

    # only c quark
    ytc1 = np.loadtxt('data/training data/results/sigma_r_training_300_c.dat')

    parameter_samples = ps1
    y_trainingli = ytli1
    y_trainingc = ytc1

    light_data = sef.load_light_data()

    # exp for experimental data
    Q2li, xli, yli, sigma_r_expli = light_data[:, 0:4].T
    tot_no_proc = light_data[:, -8] * 0.01 * sigma_r_expli  # tot_no_proc
    procedural = (light_data[:, -7:].T * 0.01 * sigma_r_expli).T
    sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))

    # c quark
    cdata = sef.load_c_data()

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
        cov_li = sef.make_cov(uncorrli, betali)
        cov_c = sef.make_cov(uncorrc, betac)
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
            zfname = zfname + ".pdf"

            print("Total emulator")
            ztitle = "GBW+charm"
            pred = emulator_draw(test_samples)
            pred_std = emulator_std(test_samples)
            baf.z_score(pred, pred_std, model, zoom=z_zoom, save_fig=z_save_fig, fname=zfname,
                        title=ztitle)

            print("Light quark emulator")
            ztitle = r"$\sigma_r \mathrm{(light \, quark)}$"
            pred = sample_y_li(test_samples)
            pred_std = emul_std_li(test_samples)
            baf.z_score(pred, pred_std, modelli, zoom=z_zoom,
                        title=ztitle)

            print("Charm quark emulator")
            ztitle = r"$\sigma_r \mathrm{(charm \, quark)}$"
            pred = sample_y_c(test_samples)
            pred_std = emul_std_c(test_samples)
            baf.z_score(pred, pred_std, modelc, zoom=z_zoom,
                        title=ztitle)

            if only_z: return

    samples = None
    if loadMCMC:
        if np.size(fname) > 1:
            samples = []
            for fname in fname:
                samples += [np.loadtxt(fname)]
        else: samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator. No sampling.")
    else:
        # import emcee.moves
        # moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        moves = None
        samples = baf.start_sampling(par_limits, labels, log_prob_emulator, saveMCMC, fname, nwalkers,
                                     nwalks, burn, walkers_par_labels=labels, moves=moves, flat=flat)

    # Plots posterior and takes plotted axis limits.
    if samples is not None:
        fig, post_limits = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)

    if more_plots:
        Dli = sigma_r_expli
        Dc = sigma_r_expc

        sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))
        sigma_r_errc = np.sqrt(uncorrc**2 + sys_erc**2)
        cov_li = sef.make_cov(uncorrli, betali)
        cov_c = sef.make_cov(uncorrc, betac)

        emuli = lambda theta: emulatorli(theta)[0]
        emuc = lambda theta: emulatorc(theta)[0]
        chi2li = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, ystd=sigma_r_errli, model_error=False)
        chi2licov = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, cov_y=cov_li, model_error=False)

        chi2c = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, ystd=sigma_r_errc, model_error=False)
        chi2ccov = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, cov_y=cov_c, model_error=False)

        chi2_tot = lambda theta: chi2li(theta) + chi2c(theta)
        chi2_tot_cov = lambda theta: chi2licov(theta) + chi2ccov(theta)

        theta_map = [20.99578715, 0.28233806, 0.96061127, 1.71801509]  # baf.get_MAP(samples[0], chi2_tot)
        # theta_map = np.quantile(samples[0], 0.5, axis=0)
        theta_map_cov = [18.23953692, 0.292427, 1.1504932, 1.97964395]  # baf.get_MAP(samples[1], chi2_tot_cov)
        # theta_map_cov = np.quantile(samples[1], 0.5, axis=0)
        print("MAP: ", theta_map)
        print("chi2/Ndof (MAP): ", chi2_tot(theta_map) / (Dc.shape[0] + Dli.shape[0] - 4))

        print("MAP for cov: ", theta_map_cov)
        print("chi2/Ndof wCov (MAP): ", chi2_tot_cov(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 4))

        print("MAP cov chi no cov: ", theta_map_cov)
        print("chi2/Ndof (MAP): ", chi2_tot(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 5))

        print("MAP chi wcov: ", theta_map)
        print("chi2/Ndof wCov (MAP): ", chi2_tot_cov(theta_map) / (Dc.shape[0] + Dli.shape[0] - 5))


def main_C_gamma(saveMCMC=False, loadMCMC=False, fname=None,
                 save_emulator=False, load_emulator=False,
                 pcacomps=5, n_restarts=1, extra_std=0,
                 nwalkers=100, nwalks=500, burn=200, flat=False,
                 zoom=None, plot_save=False, plot_fname=None,
                 zscore=False, z_zoom=False, only_z=False,
                 z_save_fig=False, zfname='z_score.png',
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
    fname : string or list of strings, optional
        The name (names) of the file (files) where the samples are loaded from. The default is None.
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
    par_limits = sef.parameter_limits(5)
    labels = sef.parameter_names(5)

    light_data = sef.load_light_data()

    # exp for experimental data
    Q2li, xli, yli, sigma_r_expli = light_data[:, 0:4].T
    tot_no_proc = light_data[:, -8] * 0.01 * sigma_r_expli  # tot_no_proc
    procedural = (light_data[:, -7:].T * 0.01 * sigma_r_expli).T
    # procedural = np.zeros_like(procedural)
    sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))

    # c quark
    cdata = sef.load_c_data()

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

    ps1 = np.loadtxt('data/training data/parameters/lhc_samples_200_5par.dat')

    ytli1 = np.loadtxt('data/training data/results/sigma_r_training_200_c_light_5par.dat')

    ytc1 = np.loadtxt('data/training data/results/sigma_r_training_200_c_5par.dat')

    parameter_samples = ps1  # np.row_stack((ps1, ps2, ps289))
    y_trainingli = ytli1  # np.row_stack((ytli1, ytli2, ytli289))
    y_trainingc = ytc1  # np.row_stack((ytc1, ytc2, ytc289))

    betali = (light_data[:, 6:-9].T * 0.01 * sigma_r_expli).T
    statli, uncli = light_data[:, 4:6].T * 0.01 * sigma_r_expli
    uncorrli = np.sqrt(statli**2 + uncli**2 + np.sum(procedural**2, axis=1))
    if cov:
        cov_li = sef.make_cov(uncorrli, betali)
        cov_c = sef.make_cov(uncorrc, betac)
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
            zfname = zfname + ".pdf"

            print("Total emulator")
            ztitle = r"GBW$^\gamma$"
            pred = emulator_draw(test_samples)
            pred_std = emulator_std(test_samples)
            baf.z_score(pred, pred_std, model, zoom=z_zoom, save_fig=z_save_fig, fname=zfname,
                        title=ztitle)

            print("Light quark emulator")
            ztitle = r"$\sigma_r \mathrm{(light \, quark)}$"
            pred = sample_y_li(test_samples)
            pred_std = emul_std_li(test_samples)
            baf.z_score(pred, pred_std, modelli, zoom=z_zoom,
                        title=ztitle)

            print("Charm quark emulator")
            ztitle = r"$\sigma_r \mathrm{(charm \, quark)}$"
            pred = sample_y_c(test_samples)
            pred_std = emul_std_c(test_samples)
            baf.z_score(pred, pred_std, modelc, zoom=z_zoom,
                        title=ztitle)
            if only_z: return

    samples = None
    if loadMCMC:
        if np.size(fname) > 1:
            samples = []
            for fname in fname:
                samples += [np.loadtxt(fname)]
        else: samples = np.loadtxt(fname)
    elif not loadMCMC and not create_emulator:
        print("No emulator nor loading samples.")
    else:
        # import emcee.moves
        # moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        moves = None
        samples = baf.start_sampling(par_limits, labels, log_prob_emulator, saveMCMC, fname, nwalkers,
                                     nwalks, burn, walkers_par_labels=labels, moves=moves, flat=flat)

    # Plots posterior.
    if samples is not None:
        fig, post_limits = baf.plotting(samples, par_limits, labels, zoom, save=plot_save, fname=plot_fname)

    if more_plots:
        Dli = sigma_r_expli
        Dc = sigma_r_expc

        sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))
        sigma_r_errc = np.sqrt(uncorrc**2 + sys_erc**2)
        cov_li = sef.make_cov(uncorrli, betali)
        cov_c = sef.make_cov(uncorrc, betac)

        emuli = lambda theta: emulatorli(theta)[0]
        emuc = lambda theta: emulatorc(theta)[0]
        chi2li = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, ystd=sigma_r_errli, model_error=False)  # / (Dli.shape[0] - 5)
        chi2licov = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, cov_y=cov_li, model_error=False)  # / (Dli.shape[0] - 5)

        chi2c = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, ystd=sigma_r_errc, model_error=False)  # / (Dc.shape[0] - 5)
        chi2ccov = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, cov_y=cov_c, model_error=False)  # / (Dc.shape[0] - 5)

        chi2_tot = lambda theta: chi2li(theta) + chi2c(theta)
        chi2_tot_cov = lambda theta: chi2licov(theta) + chi2ccov(theta)

        theta_map = [17.57815724, 0.27390233, 1.25629047, 1.07392097, 1.72693998]  # baf.get_MAP(samples[0], chi2_tot)
        # theta_map = np.quantile(samples[0], 0.5, axis=0)
        theta_map_cov = [15.14862014, 0.27829771, 1.54891792, 1.09007943, 1.99827364]  # baf.get_MAP(samples[1], chi2_tot_cov)
        # theta_map_cov = np.quantile(samples[1], 0.5, axis=0)

        print("MAP: ", theta_map)
        print("chi2/Ndof (MAP): ", chi2_tot(theta_map) / (Dc.shape[0] + Dli.shape[0] - 5))

        print("MAP for cov: ", theta_map_cov)
        print("chi2/Ndof wCov (MAP): ", chi2_tot_cov(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 5))

        print("MAP cov chi no cov: ", theta_map_cov)
        print("chi2/Ndof (MAP): ", chi2_tot(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 5))

        print("MAP chi wcov: ", theta_map)
        print("chi2/Ndof wCov (MAP): ", chi2_tot_cov(theta_map) / (Dc.shape[0] + Dli.shape[0] - 5))


if __name__ == '__main__':
    main_noC(saveMCMC=0, loadMCMC=1, fname=('data/MCMC/MCMC_noC.dat', 'data/MCMC/MCMC_noC_cov.dat'),
             save_emulator=0, load_emulator=1,
             pcacomps=10, n_restarts=10,  # extra_std=[0.00035],
             nwalkers=200, nwalks=1000, burn=500, flat=True,
             zoom='auto', z_save_fig=False,
             zscore=1, only_z=0,
             create_emulator=1, emu_std=0, emu_cov=1, cov=1,
             more_plots=True,
             plot_save=False, plot_fname='light_posterior.pdf')

    # main_C(saveMCMC=0, loadMCMC=1, fname=('data/MCMC/MCMC_wC.dat', 'data/MCMC/MCMC_wC_cov.dat'),
    #        save_emulator=0, load_emulator=1,
    #        pcacomps=10, n_restarts=10,  # extra_std=[[0.00025], [0.00035]],
    #        nwalkers=200, nwalks=1000, burn=500, flat=True,
    #        zoom='auto', z_save_fig=False,
    #        zscore=1, only_z=0,
    #        create_emulator=1, emu_std=0, emu_cov=1, cov=1,
    #        more_plots=True,
    #        plot_save=False, plot_fname='charm_posterior.pdf')

    # main_C_gamma(saveMCMC=0, loadMCMC=1, fname=('data/MCMC/MCMC_wCgamma.dat', 'data/MCMC/MCMC_wCgamma_cov.dat'),
    #              save_emulator=0, load_emulator=1,
    #              pcacomps=10, n_restarts=10,  # extra_std=[[0.0055], [0.0075]],
    #              nwalkers=200, nwalks=1000, burn=500, flat=True,
    #              zoom=[[15, 20], [0.26, 0.28], [1.05, 1.6], [1.03, 1.095], [1.68, 2.05]],  # plot_fname='kuvat/wC.png',
    #              zscore=1, only_z=0,  # z_save_fig=True,
    #              create_emulator=1, emu_std=0, emu_cov=1, cov=1,
    #              more_plots=True,
    #              plot_save=False, plot_fname='gamma_posterior.pdf')

    # For getting notification when done. Needs plyer module.
    while True:
        time.sleep(3)
        notification.notify(
            title="Finished executing",
            message="Successful",
        )
        time.sleep(1)
        break
