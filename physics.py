# -*- coding: utf-8 -*-
"""
Physics calculations.

Todo:
----
Tolerance check for integral accuracy.
Normalization check for woods-saxon?
Tests?

@author: Mikko Artturi Karhunen
"""

import numpy as np
from scipy import integrate
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


Nc = 3                              # QCD colors quantity
alpha_em = 1 / 137                    # Fine structure constant
rmax = 50  # GeV^-1
bmax = 10  # GeV^-1
binf = 50  # GeV^-1
# bb = np.linspace(0, 1, 50)
ws_norm = 1
normalised_with_A = 1
quadtol = 1e-2  # How accurate scipy integrate quad calculates the integral. Larger is faster.


def normalize_ws(A):
    """
    Normalize the int_woods_saxon function when A nucleons are used.

    This should be used before int_woods_saxon function calls.

    Parameters
    ----------
    A : int
        How many nucleons are used in the setup.

    Returns
    -------
    None.

    """
    global normalised_with_A
    if A == normalised_with_A: return
    normalised_with_A = A
    ws = lambda b: T_A(b, A=A) * b
    global ws_norm
    ws_norm = 1 / (integrate.quad(ws, 0, 100)[0] * 2 * np.pi)


def T_A(b, A):
    """
    2D Woods-Saxons distribution integrated with respect to z to 1D distribution.

    Parameters
    ----------
    b : float
        impact parameter in 1/GeV units.
    A : int
        Number of nucleons.

    Returns
    -------
    float
        Integrated Woods-Saxon distribution.

    """
    a = 0.54 * 5.068
    # r0 = 1.25 * 5.068
    R = (1.12 * A**(1 / 3) - 0.86 * A**(- 1 / 3)) * 5.068
    ws = lambda z: ws_norm / (1 + np.exp((np.sqrt(b**2 + z**2) - R) / a))
    integr = integrate.quad(ws, -50, 50, epsabs=quadtol, epsrel=quadtol)[0]
    return integr


def dipole_amplitude_nuke(b, r, x, theta, A):
    """
    Calculate dipole amplitude with integrated woods saxon distribution.

    Parameters
    ----------
    b : float
        Impact parameter in 1/GeV units.
    r : float
        The transverse size of the dipole in 1/GeV units.
    x : float
        Bjorken x
    theta : list
        The list of parameters.
    A : int
        Number of nucleons.

    Returns
    -------
    float
        Calculated dipole_amplitude.

    """
    sigma_0, lambd, x0, Q02 = theta
    Qs02 = Q02 * (x / x0)**(-lambd)
    exponent = -A * T_A(b) * (r**2 * Qs02) / 4 * sigma_0 / 2
    return 1 - np.exp(exponent)


def sigma_dip(r, x, theta, b=None, A=1, dip_theory='GBW'):
    """
    Calculate dipole cross section.

    Default GBW dipole amplitude Nr=1-np.exp(-(r^2*Qs^2)/4).

    Parameters
    ----------
    r : float
        The transverse size of the dipole in 1/GeV units.
    x : float
        Bjorken x.
    theta : list
        The list of parameters. Should be that theta = [sigma_0, lambd, x0, Qs02].
    b : float, optional
        Impact parameter in 1/GeV units. Not needed if dip_theory is not 'GBW'. The default is None.
    A : int, optional
        Number of nucleons. The default is 1.
    dip_theory : string, optional
        Determines which dipole theory model is used. The default is GBW model for only one proton.
        That case this does not need impact parameter. If set to Nuke the model includes
        whole nucleus and needs impact parameter. The default is GBW.

    Returns
    -------
    float
        Dipole cross section with given theta's parametization.

    """
    sigma_0, lambd, x0, Qs02 = theta
    Qs2 = Qs02 * (x / x0)**(-lambd)               # Saturation scale squared. Qs^2
    if dip_theory == 'Nuke':
        # nucleus
        return 2 * dipole_amplitude_nuke(b, r, x, theta, A=A)
    # GBW
    return sigma_0 * (1 - np.exp(-(r**2 * Qs2) / 4))


def sigma_T(x, Q2, theta, Zf, mf, A=1, dip_theory='GBW'):
    """
    Calculate transverse photon-proton cross section. Can also be set to photon-nucleus.

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int)
        Q^2. The virtuality of the photon or virtual photons "mass" squared. [GeV^2]
    theta : list
        The list of parameters.
    Zf : array
        The fractional electric charges of the quarks.
    mf : array
        The masses of the quarks.
    A : int, optional
        Number of nucleons. The default is 1. Not used with GBW model.
    dip_theory : string, optional
        Determines which dipole theory model is used. The default is GBW model for only one proton.
        That case this does not need impact parameter. If set to Nuke the model includes
        whole nucleus and needs impact parameter. The default is GBW.

    Returns
    -------
    float
        Transverse cross section without error.
    """
    af = lambda z: np.sqrt(Q2 * z * (1 - z) + mf**2)

    # Kovchekov book, chapter 4
    # The square of the absolute value of the transverse wave function.
    # kn() is the modified bessel function of second kind from scipy.
    # sigma^qqA_tot = sigma_dip from arxiv 1711.11360
    def Psi_T_abs_square(r, z):
        return Nc * alpha_em * Zf**2 / np.pi * (af(z)**2 * kn(1, r * af(z))**2 * (z**2 + (1 - z)**2) + mf**2 * kn(0, r * af(z))**2)

    # Double integral, z from 0 to 1, r from 0 to 50. 50 is aproximation of infinity. Speed depents on iterations (size) of the integral.
    if dip_theory == 'GBW':
        # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
        transverse = lambda r, z: np.sum(Psi_T_abs_square(r, z)) * sigma_dip(r, x, theta) * r
        return integrate.dblquad(transverse, 0, 1, 0, rmax, epsabs=quadtol, epsrel=quadtol)[0]    # [0] for only value, no error. error is assumed to be negligible.
    else:
        # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
        transverse = lambda b, r, z: np.sum(Psi_T_abs_square(r, z)) * sigma_dip(r, x, theta, b, A=A, dip_theory="Nuke") * r * b * 2 * np.pi
        b_to_inf_trans = lambda b, r, z: A * T_A(b) * np.sum(Psi_T_abs_square(r, z)) * sigma_dip(r, x, theta, b, A=A, dip_theory="GBW") * r * b * 2 * np.pi
        return (integrate.tplquad(transverse, 0, 1, 0, rmax, 0, bmax, epsabs=quadtol, epsrel=quadtol)[0]
                + integrate.tplquad(b_to_inf_trans, 0, 1, 0, rmax, bmax, binf, epsabs=quadtol, epsrel=quadtol)[0])


def sigma_L(x, Q2, theta, Zf, mf, A=1, dip_theory='GBW'):
    """
    Calculate longitudal photon-proton cross section. Can also be set to photon-nucleus.

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int)
        Q^2. The virtuality of the photon or virtual photons "mass" squared. [GeV^2]
    theta : list
        The list of parameters.
    Zf : array
        The fractional electric charges of the quarks.
    mf : array
        The masses of the quarks.
    A : int, optional
        Number of nucleons. The default is 1. Not used with GBW model.
    dip_theory : string, optional
        Determines which dipole theory model is used. The default is GBW model for only one proton.
        That case this does not need impact parameter. If set to Nuke the model includes
        whole nucleus and needs impact parameter. The default is GBW.

    Returns
    -------
    float
        Longitudal cross section without error.

    """
    af = lambda z: np.sqrt(Q2 * z * (1 - z) + mf**2)

    # Kovchekov book, chapter 4
    # The square of the absolute value of the transverse wave function. kn() is the modified bessel function of second kind from scipy. sigma^qqA_tot = sigma_dip from arxiv 1711.11360
    Psi_L_abs_square = lambda r, z: Nc * alpha_em * Zf**2 / np.pi * 4 * Q2 * kn(0, r * af(z))**2

    if dip_theory == 'GBW':
        # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
        longitudal = lambda r, z: z**2 * (1 - z)**2 * np.sum(Psi_L_abs_square(r, z)) * sigma_dip(r, x, theta) * r

        # Double integral, z from 0 to 1, r from 0 to 50. 50 is aproximation of infinity. Speed depents on iterations (size) of the integral.
        return integrate.dblquad(longitudal, 0, 1, 0, rmax, epsabs=quadtol, epsrel=quadtol)[0]    # [0] for only value, no error. error is assumed to be negligible.
    # The function for integral. \int dr^2 = \int dr r \int dtheta = \int dr r 2pi
    longitudal = lambda b, r, z: z**2 * (1 - z)**2 * np.sum(Psi_L_abs_square(r, z)) * sigma_dip(r, x, theta, b, A=A, dip_theory="Nuke") * r * b * 2 * np.pi
    b_to_inf_longitud = lambda b, r, z: A * T_A(b) * z**2 * (1 - z)**2 * np.sum(Psi_L_abs_square(r, z)) * sigma_dip(r, x, theta, b, A=A, dip_theory="GBW") * r * b * 2 * np.pi
    # Double integral, z from 0 to 1, r from 0 to 50. 50 is aproximation of infinity. Speed depents on iterations (size) of the integral.
    return (integrate.tplquad(longitudal, 0, 1, 0, rmax, 0, bmax, epsabs=quadtol, epsrel=quadtol)[0]
            + integrate.tplquad(b_to_inf_longitud, 0, 1, 0, rmax, bmax, binf, epsabs=quadtol, epsrel=quadtol)[0])


def F2(x, Q2, sigma0, lambd, Q02, A=1, dip_theory='GBW'):
    """
    Calculate the value of the DIS proton structurefunction F2(x,Q^2) with given parameter vector theta.

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int) [GeV^2]
        Q^2. The virtuality of the photon or virtual photons "mass" squared. Units [GeV^2]
    sigma0 : float [mb]
        sigma_0. Proportional to the transverse size of the proton.
    lambd : float
        The evolution speed of the saturation scale.
    Q02 : float [GeV^2]
        Q_0^2. Related to saturation scale.
    A : int, optional
        Number of nucleons. The default is 1. Not used with GBW model.
    dip_theory : string, optional
        Determines which dipole theory model is used. The default is GBW model for only one proton.
        That case this does not need impact parameter. If set to Nuke the model includes
        whole nucleus and needs impact parameter. The default is GBW.

    Returns
    -------
    float or array
        Proton structurefunction's value.

    """
    if dip_theory != 'GBW':
        # Initializing nucleus calculations.
        normalize_ws(A)
        bb = np.linspace(0, 100, 100)
        TA_bb = [T_A(b, A) for b in bb]
        global T_A
        T_A = lambda b: interp1d(bb, TA_bb, bounds_error=False, fill_value=0)(b)
        normtest = integrate.quad(lambda b: T_A(b) * b * 2 * np.pi, 0, np.inf)[0]
        assert np.isclose(normtest, 1.0, 1e-2, 1e-2), "normalisation failed for T_A: n=" + str(normtest)

        global bmax
        bmax = fsolve(lambda b: A * T_A(b) * sigma0 * 2.56819 / 2 - 1, 7)[0]
        assert binf > bmax, "bmax larger than binf. bmax: " + str(bmax)

    x0 = 2.24 * 10**(-4)
    Zf = np.array([2 / 3, -1 / 3, -1 / 3])    # Up, down, strange quark fractional charges.
    mf = 0.14                                 # GeV Light quark masses
    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02]
    F = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zf, mf, A=A, dip_theory=dip_theory) + sigma_L(x, Q2, theta, Zf, mf, A=A, dip_theory=dip_theory))
    return F


def sigma_r(x, Q2, y, sigma0, lambd, Q02, mc=None):
    """
    Calculate reduced cross section with or without charm quark. Only for e^+ p collisions.

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int) [GeV^2]
        Q^2. The virtuality of the photon or virtual photons "mass" squared.
    y : float
        Elasticity.
    sigma0 : float [mb]
        sigma_0. Proportional to the transverse size of the proton.
    lambd : float
        The evolution speed of the saturation scale.
    Q02 : float [GeV^2]
        Q_0^2. Related to saturation scale.
    mc : float, optional
        The charm quark's mass [GeV]. The default is None.

    Returns
    -------
    sig_r : float or array
        Reduced cross section for photon-proton collision.

    """
    x0 = 2.24 * 10**(-4)
    mf = 0.14                           # GeV Light quark masses
    # If no mc, only light quarks are accounted.
    if mc is None:
        Zf = np.array([2 / 3, -1 / 3, -1 / 3])    # Fractional charges for up, down, strange quarks.
        # x = np.array([x, x, x, ])
    else:
        Zf = np.array([2 / 3, -1 / 3, -1 / 3, 2 / 3])   # Up, down, strange, charm quark charges. Unit: 1e
        mf = np.array([mf, mf, mf, mc])         # Light quark masses and charm mass

    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02]
    Yplus = 1 + (1 - y)**2
    sig_r = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zf, mf) + sigma_L(x, Q2, theta, Zf, mf) * (1 - y**2 / Yplus))
    return sig_r


def sigma_r_c(x, Q2, y, sigma0, lambd, Q02, mc):
    """
    Calculate reduced cross section. Only c-quark contribution. Only for e^+ p collisions.

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int) [GeV^2]
        Q^2. The virtuality of the photon or virtual photons "mass" squared.
    y : float
        Elasticity.
    sigma0 : float [mb]
        sigma_0. Proportional to the transverse size of the proton.
    lambd : float
        The evolution speed of the saturation scale.
    Q02 : float [GeV^2]
        Q_0^2. Related to saturation scale.
    mc : float
        The charm quark's mass [GeV].

    Returns
    -------
    sig_r : float or array
        Reduced cross section with only c-quark contribution.

    """
    x0 = 2.24 * 10**(-4)
    Zc = 2 / 3    # Fractional charm quark charge.
    # sigma0 from mb to Gev^-2
    theta = [sigma0 * 2.56819, lambd, x0, Q02]
    Yplus = 1 + (1 - y)**2
    sig_r = Q2 / (4 * np.pi**2 * alpha_em) * (sigma_T(x, Q2, theta, Zc, mc) + sigma_L(x, Q2, theta, Zc, mc) * (1 - y**2 / Yplus))
    return sig_r


def ratio_R(x, Q2, sigma0, lambd, Q02, A):
    """
    Calculate nucleus to proton times nucleons ratio. F2_nucleus(A)/(F2_proton * A).

    Parameters
    ----------
    x : float
        Bjorken x.
    Q2 : float (or int) [GeV^2]
        Q^2. The virtuality of the photon or virtual photons "mass" squared. Units [GeV^2]
    sigma0 : float [mb]
        sigma_0. Proportional to the transverse size of the proton.
    lambd : float
        The evolution speed of the saturation scale.
    Q02 : float [GeV^2]
        Q_0^2. Related to saturation scale.
    A : int
        Number of nucleons in the nucleus.

    Returns
    -------
    R : float
        The ratio of the F2(nucleus of A nucleons)/(F2(proton)*A).

    """
    R = F2(x, Q2, sigma0, lambd, Q02, A=A, dip_theory='Nuke') / F2(x, Q2, sigma0, lambd, Q02, A=1, dip_theory='GBW') / A
    return R


if __name__ == '__main__':

    # # This is for testing
    # # sigma_0, lambd, x0, Qs02 = theta
    # theta1 = [16 * 2.56819, 0.3, 2.24 * 10**(-4), 2]
    # testingfunc = lambda b: sigma_dip(0.0001, 0.01, theta1, b, A, dip_theory='Nuke') * b * 2 * np.pi
    # test2 = integrate.quad(testingfunc, 0, np.inf)[0]
    # test3 = sigma_dip(0.0001, 0.01, theta1)
    # print(test2 / test3)
    # # print(sigma_r(0.01, 2, 0.01, 16, 0.3, 1))
    # # x, Q2, sigma0, lambd, Q02,
    # A = 197
    # parametization = [0.01, 40, 16, 0.3, 2]
    # print("A =", A)
    # print(ratio_R(*parametization, A=A))
    # print(F2(*parametization, A=A, dip_theory='Nuke') / F2(*parametization, A=1, dip_theory='GBW') / A)

    # import matplotlib.pyplot as plt
    # rvals = np.geomspace(1e-3, 100)
    # plt.plot(rvals, [integrate.quad(lambda b: 2 * 2.0 * np.pi * b * dipole_amplitude_nuke(b, r, 1e-3, A=197, theta=theta1), 0, bmax)[0]
    #                   + integrate.quad(lambda b: A * T_A(b) * 2.0 * np.pi * b * sigma_dip(r, 1e-3, theta=theta1), bmax, 200)[0] for r in rvals], label="Nuke")

    # plt.plot(rvals, 197 * np.array([sigma_dip(r, 1e-3, theta=theta1) for r in rvals]), linestyle="dashed", label="A proton")
    # plt.xscale("log")
    # # plt.yscale("log")
    # leg = plt.legend(loc="upper left")
    # plt.xlabel(r"$r$")
    # plt.ylabel(r"$2\int d^2b N(r,b)$")
    # plt.tight_layout()
    # plt.savefig("./plots/proton_vs_nuke_bint.pdf")

    # print(sigma_r(0.01, 2, 0.01, 16, 0.3, 1))
