# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:16:12 2023

@author: Omistaja
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_R(calculated_R, x, Q2, title=''):

    # plt.savefig("./proton_vs_nuke_bint.pdf")

    # plt.figure(figsize=(8, 6))
    plt.title(title)

    def apu(calculated_R, labe):
        mean = np.mean(calculated_R, axis=0)
        std = np.std(calculated_R, axis=0)

        # plt.errorbar(x, sigma_r_exp, sigma_r_err, fmt='*', markersize=1.5, elinewidth=0.5, capsize=1, color='k', label="data", alpha=0.4)
        plt.plot(Q2, mean, label=labe)
        plt.fill_between(Q2, mean + 2 * std, mean - 2 * std, alpha=0.2, label=r"2$\sigma$ margin")

    R = np.loadtxt("data/other/ratio_R_light_posterior_Q2_20.dat")
    apu(R, "R ratio")
    R = np.loadtxt("data/other/ratio_R_light_posterior_cov_Q2_20.dat")
    apu(R, "R ratio with covariance accounted")
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend(loc="upper left")
    # Creating a information box
    props = dict(boxstyle='round', fc='white', facecolor='white', alpha=0.7, ec='silver')
    plt.text(0.83, 0.98, r"$x_{Bj}$" + f" = {x}", bbox=props)
    plt.xlabel(r"$Q^2$ [$GeV^2$]")
    plt.ylabel(r"$R_{eA}$")
    plt.tight_layout()
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()
    plt.savefig("nuclear_suppression.pdf")


if __name__ == '__main__':
    Q2 = np.geomspace(1, 1000, 20)
    x = 0.001
    # samps = np.loadtxt("data/other/light_posterior_hundred_samples.dat")
    R = np.loadtxt("data/other/ratio_R_light_posterior_Q2_20.dat")
    title = ""  # '100 random samples average R from posterior'
    plot_R(R, x, Q2, title)
