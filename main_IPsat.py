# -*- coding: utf-8 -*-
"""
The main file for the bayesian analysis IPsat model setup.

@author: Mikko Artturi Karhunen
"""

import pickle
import numpy as np
# from plyer import notification  # For getting notification, when the run is finished.
import time
import probability_formulas as prob_calc
import bayesian_analysis_functions as baf
import setup_functions as sef

# TODO: Maybe load data straigth from hera?
# import urllib.request
# import requests
# target_url = "https://www.desy.de/h1zeus/herapdf20/HERA1+2_NCep_920.dat"
# data = urllib.request.urlopen(target_url)  # it's a file like object and works just like a file
# # data = requests.get(target_url)
# # data = data.text
# for line in data:  # files are iterable
#     print(line)


def cut_hera1(data):
    """
    Cut data with limits 1.49< Q2 <50.1 and x < 0.01.

    This is used for hera1 data from 2013 publication.

    Parameters
    ----------
    data : array
        Hera data to be cut.

    Returns
    -------
    array
        The cut hera data.

    """
    return data[(data[:, 0] > 1.49) & (data[:, 0] < 50.1) & (data[:, 1] < 0.01) & (data[:, 1] > 1e-99)]


def mainIPsat(saveMCMC=False, loadMCMC=False, fname=None,
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
    names = [r"$m_c$ [GeV]", "C", r"$\lambda_g$", r"$A_g$"]
    limits = np.array([[1.1, 1.8], [0.5, 10], [0.01, 0.5], [0.2, 10]])
    par_limits = limits
    labels = names

    # hera 2013 data
    # datali = cut_hera1(np.loadtxt("IPsat/data/hera_combined_sigmar.txt"))
    # datac = cut_hera1(np.loadtxt("IPsat/data/hera_combined_sigmar_cc.txt"))

    parameter_samples = np.loadtxt("IPsat/data/training/parameters/param_list_600_4.dat")
    traindata = np.loadtxt("IPsat/data/training/results/sigma_r_hera_II_600.dat")
    y_trainingli = traindata[:, :-34]  # This is to separate light and charm training data.
    y_trainingc = traindata[:, -34:]

    # Load experiemtnal data
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
        noise_level_bounds = [1e-10, 1e1]
        length_scale_bounds = [1e-3, 1000]
        if not load_emulator:
            pca_gpeli = baf.make_emulator(parameter_samples, y_trainingli, par_limits, pcacomps,
                                          whiten, n_restarts, length_scale_bounds=length_scale_bounds,
                                          noise_level_bounds=noise_level_bounds)
            pca_gpec = baf.make_emulator(parameter_samples, y_trainingc, par_limits, pcacomps,
                                         whiten, n_restarts, length_scale_bounds=length_scale_bounds,
                                         noise_level_bounds=noise_level_bounds)
            if save_emulator:
                file = open("emulators/emulator_ipsat_li", "wb")  # trained for hera2 data
                pickle.dump(pca_gpeli, file)
                file = open("emulators/emulator_ipsat_c", "wb")
                pickle.dump(pca_gpec, file)
        else:
            file = open("emulators/emulator_ipsat_li", "rb")  # trained for hera2 data
            pca_gpeli = pickle.load(file)
            file = open("emulators/emulator_ipsat_c", "rb")
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
            # sample_size = 200
            test_samples = np.loadtxt("IPsat/data/testing/param_test_200_par4.dat")
            model = np.loadtxt("IPsat/data/testing/sigma_r_hera_II_test_200.dat")
            modelli = model[:, :-34]
            modelc = model[:, -34:]

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
            ztitle = "IPsat"
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
                                     nwalks, burn, moves=moves, flat=flat)
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
            sef.samples_plot(sampless[1], 5000, post_limits, emulatorli, xli, Q2li, yli, sigma_r_expli, sigma_r_errli)

        emuli = lambda theta: emulatorli(theta)[0]
        emuc = lambda theta: emulatorc(theta)[0]
        chi2li = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, ystd=sigma_r_errli, model_error=False)  # / (Dli.shape[0] - 5)
        chi2licov = lambda theta: prob_calc.log_likelihood(Dli, emuli, theta, cov_y=cov_li, model_error=False)  # / (Dli.shape[0] - 5)

        chi2c = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, ystd=sigma_r_errc, model_error=False)  # / (Dc.shape[0] - 5)
        chi2ccov = lambda theta: prob_calc.log_likelihood(Dc, emuc, theta, cov_y=cov_c, model_error=False)  # / (Dc.shape[0] - 5)

        chi2_tot = lambda theta: chi2li(theta) + chi2c(theta)
        chi2_tot_cov = lambda theta: chi2licov(theta) + chi2ccov(theta)

        theta_map = [1.33006287, 2.01625159, 0.08764014, 2.14053997]  # baf.get_MAP(samples[0], chi2_tot)
        # theta_map = np.quantile(samples[0], 0.5, axis=0)
        theta_map_cov = [1.37190592, 2.54735337, 0.06793598, 2.27400732]  # baf.get_MAP(samples[1], chi2_tot_cov)
        # theta_map_cov = np.quantile(samples[1], 0.5, axis=0)

        print("MAP: ", theta_map)
        print("chi2/Ndof (MAP): ", chi2_tot(theta_map) / (Dc.shape[0] + Dli.shape[0] - 4))

        print("MAP for cov: ", theta_map_cov)
        print("chi2/Ndof wCov (MAP): ", chi2_tot_cov(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 4))

        print("MAP cov: ", theta_map_cov)
        print("chi2/Ndof (MAP_cov): ", chi2_tot(theta_map_cov) / (Dc.shape[0] + Dli.shape[0] - 4))

        print("MAP: ", theta_map)
        print("chi2_cov/Ndof wCov (MAP): ", chi2_tot_cov(theta_map) / (Dc.shape[0] + Dli.shape[0] - 4))

        Tli = emulatorli(theta_map)[0]
        Tcovli = emulatorli(theta_map_cov)[0]

        sef.more_plots(sigma_r_expli, Tli, Tcovli, betali, uncorrli, Q2li, xli)


if __name__ == '__main__':
    mainIPsat(saveMCMC=0, loadMCMC=1, fname=("IPsat/data/MCMC/MCMC_IPsat.dat", "IPsat/data/MCMC/MCMC_IPsat_cov.dat"),
              save_emulator=0, load_emulator=1,
              pcacomps=10, n_restarts=10,  # extra_std=[[0.00085], [0.00055]],
              nwalkers=200, nwalks=1000, burn=500, flat=True,
              zoom='auto',  # plot_fname='kuvat/wC_temp.png',
              zscore=1, only_z=0, z_save_fig=0,
              create_emulator=1, emu_std=0, emu_cov=1, cov=1,
              more_plots=True,
              plot_save=False, plot_fname='IPsat_posterior_cov.pdf')

    # For getting notification when done. Needs plyer module.
    # while True:
    #     time.sleep(3)
    #     notification.notify(
    #         title="Finished executing",
    #         message="Successful",
    #     )
    #     time.sleep(1)
    #     break
