# -*- coding: utf-8 -*-
"""
The physics calculations but with gamma parameter included. This is an older version.

@author: Mikko Karhunen
"""
from scipy.special import kn
from scipy import integrate
import numpy as np


Nc = 3                              # QCD colors quantity
alpha_em = 1 / 137                    # Fine structure constant


def sigma_dip(x, theta):
    '''
    Calculates dipole cross section. Dipole amplitude Nr=1-np.exp(-(r^2*Qs^2)/4).
    arxiv 1711.11360

    Arguments:
    x: Bjorken x
    theta: parameter vector

    Returns:
    Calculated dipole cross section with theta's parameters
    '''
    sigma_0, lambd, x0, Q02, gamma = theta
    Qs2 = Q02 * (x / x0)**(-lambd)               # Saturation scale squared. Qs^2
    # Nr = lambda r: 1 - np.exp(-(r**2 * Qs2) / 4)   # The dipole-proton amplitude
    Nr = lambda r: 1 - np.exp(-((r**2 * Qs2)**gamma) / 4)   # The dipole-proton amplitude
    return lambda r: sigma_0 * Nr(r)


def sigma_T(x, Q2, theta, Zf, mf):
    '''
    Calculate transverse gamma*-proton cross section. 

    Arguments:
    x: Bjorken x
    Q2: Q^2 virtual photons "mass" squared
    theta: parameter vector

    Returns:
    Transverse cross section without error.
    '''
    af = lambda z: np.sqrt(Q2 * z * (1 - z) + mf**2)

    # Kovchekov book, chapter 4
    # The square of the absolute value of the transverse wave function. kn() is the modified bessel function of second kind from scipy. sigma^qqA_tot = sigma_dip from arxiv 1711.11360
    Psi_T_abs_square = lambda r, z: 2 * Nc * np.sum(alpha_em * Zf**2 / np.pi * (af(z)**2 * kn(1, r * af(z))**2 * (z**2 + (1 - z)**2) + mf**2 * kn(0, r * af(z))**2))

    # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
    transverse = lambda r, z: 1 / 2 * Psi_T_abs_square(r, z) * sigma_dip(x, theta)(r) * r

    # Double integral, z from 0 to 1, r from 0 to 50. 50 is aproximation of infinity. Speed depents on iterations (size) of the integral.
    return integrate.dblquad(transverse, 0, 1, 0, 50)[0]    # [0] for only value, no error. error is assumed to be negligible.


def sigma_L(x, Q2, theta, Zf, mf):
    '''
    Calculate longitudal gamma*-proton cross section.

    Arguments:
    x: Bjorken x
    Q2: Q^2 virtual photons "mass" squared
    theta: parameter vector

    Returns:
    Longitudal cross section without error.
    '''
    af = lambda z: np.sqrt(Q2 * z * (1 - z) + mf**2)

    # Kovchekov book, chapter 4
    # The square of the absolute value of the transverse wave function. kn() is the modified bessel function of second kind from scipy. sigma^qqA_tot = sigma_dip from arxiv 1711.11360
    Psi_L_abs_square = lambda r, z: 2 * Nc * np.sum(alpha_em * Zf**2 / np.pi * 4 * Q2 * kn(0, r * af(z))**2)

    # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
    longitudal = lambda r, z: 1 / 2 * z**2 * (1 - z)**2 * Psi_L_abs_square(r, z) * sigma_dip(x, theta)(r) * r

    # Double integral, z from 0 to 1, r from 0 to 50. 50 is aproximation of infinity. Speed depents on iterations (size) of the integral.
    return integrate.dblquad(longitudal, 0, 1, 0, 50)[0]    # [0] for only value, no error. error is assumed to be negligible.


def F2(x, Q2, sigma0, lambd, Q02, gamma, x0=2.24 * 10**(-4)):
    '''
    Calculates the value of the DIS proton structurefunction F2(x,Q^2) with given parameter vector theta.

    Arguments:
    x: Bjorken x
    Q2: Q^2 virtual photons "mass" squared. [GeV^2]
    sigma0: \sigma_0 [mb]
    lambd: \lambda
    Q02: Q0^2, [GeV^2]

    Returns:
    Proton structurefunctions values in 2D array.
    '''
    Zf = np.array([2 / 3, -1 / 3, -1 / 3])    # Up, down, strange quark charges. Unit: 1e
    mf = 0.14                           # GeV Light quark masses
    # maybe mf = np.array([mf, mf, mf])
    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02, gamma]
    F = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zf, mf) + sigma_L(x, Q2, theta, Zf, mf))
    return F  # or maybe np.ravel(F)


def sigma_r(x, Q2, y, sigma0, lambd, Q02, gamma, mc=None):
    '''
    Reduced cross section with or without charm quark.

    Arguments:
    x: Bjorken x
    Q2: Q^2 virtual photons "mass" squared. [GeV^2]
    y: y from experimental data
    sigma0: \sigma_0 [mb]
    lambd: \lambda
    Q02: Q0^2, [GeV^2]
    mc: Charm quark mass [GeV]. The Default is None.

    Returns:
    Proton structurefunctions values in 2D array.
    '''
    x0 = 2.24 * 10**(-4)
    mf = 0.14                           # GeV Light quark masses

    # If no mc, only light quarks are accounted.
    if mc is None:
        Zf = np.array([2 / 3, -1 / 3, -1 / 3])    # Charger for up, down, strange. Unit: 1e
        mf = np.array([mf, mf, mf])
    else:
        Zf = np.array([2 / 3, -1 / 3, -1 / 3, 2 / 3])   # Up, down, strange, charm quark charges. Unit: 1e
        mf = np.array([mf, mf, mf, mc])         # Light quark masses and charm mass

    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02, gamma]
    Yplus = 1 + (1 - y)**2
    sig_r = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zf, mf) + sigma_L(x, Q2, theta, Zf, mf) * (1 - y**2 / Yplus))
    return sig_r


def sigma_r_c(x, Q2, y, sigma0, lambd, Q02, gamma, mc):
    '''
    Reduced cross section. Only c-quark contribution. Only for e^+ p collisions. +charge

    Arguments:
    x: Bjorken x
    Q2: Q^2 virtual photons "mass" squared. [GeV^2]
    y: y from experimental data
    sigma0: \sigma_0 [mb]
    lambd: \lambda
    Q02: Q0^2, [GeV^2]
    mc: Charm quark mass [GeV]. The Default is None.

    Returns:
    Proton structurefunctions values in 2D array.
    '''
    x0 = 2.24 * 10**(-4)
    Zc = 2 / 3    # Charm quark charge. Unit: 1e
    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02, gamma]
    Yplus = 1 + (1 - y)**2
    sig_r = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zc, mc) + sigma_L(x, Q2, theta, Zc, mc) * (1 - y**2 / Yplus))
    return sig_r
