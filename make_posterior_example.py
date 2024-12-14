# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:12:12 2024
A posterior creation example using IPsat training data and HERA experimental 
data.

@author: Mikko
"""

import numpy as np
import bayesian_analysis_functions as baf
import setup_functions as sef


par_lims = np.array([[1.1, 1.8], [0.5, 10], [0.01, 0.5], [0.2, 10]])
par_labs = [r"$m_c$ [GeV]", "C", r"$\lambda_g$", r"$A_g$"]

training_parameters = "IPsat/data/training/parameters/param_list_600_4.dat"
training_data = np.loadtxt("IPsat/data/training/results/sigma_r_hera_II_600.dat")
training_datali = training_data[:, :-34]  # This is to separate light and charm training data.
training_datac = training_data[:, -34:]
testing_parameters = "IPsat/data/testing/param_test_200_par4.dat"
testing_data = np.loadtxt("IPsat/data/testing/sigma_r_hera_II_test_200.dat")
testing_datali = testing_data[:, :-34]
testing_datac = testing_data[:, -34:]


# Load inclusive HERA experiemtnal data
light_data = sef.load_light_data()

# exp for experimental data
Q2li, xli, yli, sigma_r_expli = light_data[:, 0:4].T
tot_no_proc = light_data[:, -8] * 0.01 * sigma_r_expli  # tot_no_proc
procedural = (light_data[:, -7:].T * 0.01 * sigma_r_expli).T
# procedural = np.zeros_like(procedural)
sigma_r_errli = np.sqrt(tot_no_proc**2 + np.sum(procedural**2, axis=1))

betali = (light_data[:, 6:-9].T * 0.01 * sigma_r_expli).T
statli, uncli = light_data[:, 4:6].T * 0.01 * sigma_r_expli
uncorrli = np.sqrt(statli**2 + uncli**2 + np.sum(procedural**2, axis=1))


# c data
cdata = sef.load_c_data()

Q2c, xc, sigma_r_expc = cdata[:, 0:3].T
# yc = Q2c / (318**2 * xc)
statc, uncc = cdata[:, (3, 4)].T * 0.01 * sigma_r_expc
uncorrc = np.sqrt(statc**2 + uncc**2)
betac = (cdata[:, 5:].T * 0.01 * sigma_r_expc).T

# The uncertainty estimates and experimental data assingments
ex_datali = sigma_r_expli
ex_datac = sigma_r_expc
cov_li = sef.make_cov(uncorrli, betali)
cov_c = sef.make_cov(uncorrc, betac)


# The tuning of setup
MCMC_walkers = 25
MCMC_burn = 100
MCMC_steps = 200
npcs = 5

kwargs_PCA = {"pca_whiten": True, "pca_n_restarts": 1,
              "pca_length_scale_bounds": [0.001, 1000],
              "pca_noise_level_bounds": [1e-19, 1e1]}

baf.make_posterior(training_parameters, [training_datali, training_datac],
                   testing_parameters, [testing_datali, testing_datac],
                   [ex_datali, ex_datac], [cov_li, cov_c],
                   par_lims, par_labs, MCMC_walkers,
                   MCMC_burn, MCMC_steps, npcs,
                   # load_emulator=["emulator_1", "emulator_2"],
                   calc_zscore=True,
                   create_emulator=True,
                   # load_samples='data/MCMC/MCMC_noC_cut.dat',
                   kwargs_MCMC={"flat": False},
                   kwargs_plotting={"zoom": "auto"})
