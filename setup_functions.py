import matplotlib.pyplot as plt
import numpy as np


def nuisance_profiling(D, T, beta, uncorr):
    """
    Generate nuisance parameters and plot them.

    Parameters
    ----------
    D : array
        Experimental data.
    T : array
        Model prediction data.
    beta : array
        Systematic errors.
    uncorr : array
        Uncorrelated errors.

    Returns
    -------
    nuis : array
        nuisance parameters.

    """
    print("------")
    print("nuisance parameters.")
    N, sysN = beta.shape
    delta = np.eye(sysN)
    A = np.einsum('ik, il, i -> kl', beta, beta, 1 / uncorr**2) + delta
    A_inv = np.linalg.inv(A)
    nuis = np.einsum('i, hk, ik, i -> h', D - T, A_inv, beta, 1 / uncorr**2)

    mean = np.mean(nuis)
    std = np.std(nuis)
    plt.figure()
    plt.hist(nuis, 100, density=True)  # Plot mean as a vertical line

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=0.9, alpha=0.8)
    plt.axvline(mean + std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    plt.axvline(mean - std, color='k', linestyle='dashed', linewidth=0.5, alpha=0.8)
    plt.title('Nuisance parameters. Mean: {:.3f}, Std: {:.3f}'.format(mean, std), fontsize=10)
    plt.savefig('nuis.pdf')
    plt.show()

    print("mean:")
    print(mean)
    print("std:")
    print(std)
    print("--------")
    return nuis


def plot_data(x, D, D_er=None, T=None, labels=None, title=None):
    """
    Plot data with experimental error bar against model data.

    Parameters
    ----------
    x : array
        Bjorken x from experimental data.
    D : array
        Experimental data.
    D_er : array, optional
        Error of experimental data. The default is None.
    T : array, optional
        The model data. The default is None.
    labels : string, optional
        The labels for plots. The default is None.
    title : string, optional
        The title for the plot. The default is None.

    Returns
    -------
    None.

    """
    plt.figure()
    if D_er is not None: plt.errorbar(x, D, yerr=D_er, fmt='o', markersize=1, linewidth=0.5, label=labels[0], color='black')
    if labels is not None:
        if D_er is None: plt.scatter(x, D, s=1, label=labels[0])
        plt.scatter(x, T, s=1, label=labels[1], color='red')

    else:
        if D_er is None: plt.scatter(x, D, s=1)
        plt.scatter(x, T, s=1)
    plt.xscale("log")
    # plt.ylim([0.2, 1.6])
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.ylabel(r"$\sigma_r$")
    plt.xlabel(r'$x_{Bj}$')
    plt.legend()
    plt.savefig(str.lower(title) + ".pdf")
    plt.show()


def more_plots(D, T, Tcov, beta, uncorr, Q2, x):
    """
    Generate some more plots.

    Parameters
    ----------
    D : array
        Experimental data.
    T : array
        The model data calculated with parameters which were optained without covariance accounted.
    Tcov : array
        The model calculated with parameters that was got with covariance accounted.
    beta : array
        Systematic errors of the experimental data.
    uncorr : array
        The uncorrelated uncertainty of the experimental data.
    Q2 : float (or int) [GeV^2]
        Q^2. The virtuality of the photon or virtual photons "mass" squared.
    x : float
        Bjorken x.

    Returns
    -------
    D_shifted : array
        Shifted experimental data according to optained covariance parametisation.

    """
    def plot_diff(D, D_shifted, syserr):
        """
        Plot the difference between shifted data and original data relative to systematic error.

        Parameters
        ----------
        D : array
            Original data.
        D_shifted : array
            Shifted data.
        syserr : array
            systematic error

        Returns
        -------
        None.

        """
        diff = (D - D_shifted) * 100 / D
        plt.figure()
        plt.plot(diff)
        plt.grid('True')
        plt.title("Percent difference from the original experimental sigma_r by shifting")
        plt.show()

        diffpersyst = (D - D_shifted) / syserr
        plt.figure()
        plt.hist(diffpersyst, 100)
        plt.title(r"Histogram of ($D-D_s)/\sigma_{sys}$")
        plt.savefig('hist.pdf')
        plt.show

    def plot_against_exp(x, D, syserr, D_shifted, model, model_cov, title):
        """
        Plot data and model and model with covariance parametisation to same plot.

        Parameters
        ----------
        x : array
            Bjorken x.
        D : array
            Data.
        syserr : array
            Systematic error of data
        D_shifted : array
            Shifted data.
        model : array
            The model data.
        model_cov : array
            The model data optained by parametisation gotten from covariance setup.
        title : string
            The title for the plot.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(5, 3))
        plt.xscale("log")
        # plt.ylim([0.2, 1.6])
        plt.grid(True, alpha=0.3)
        plt.ylabel(r"$\sigma_r$")
        plt.xlabel(r"$x_{Bj}$")

        plt.title(title)

        plt.errorbar(x, D, yerr=syserr, fmt='o', markersize=1, linewidth=0.5, label=r"$\sigma_r$ with sys err", color='black')
        # plt.scatter(x1, D1, s=1, label=r"$\sigma_r$", color='black')
        plt.scatter(x, D_shifted, s=1, label=r"$\sigma_r shifted$", color='red', marker="v")

        lw = 1
        plt.plot(x, model, label="Model", lw=lw)
        # plt.fill_between(x, model + 2 * model_err, model - 2 * model_err, alpha=0.2, label=r"2$\sigma$ margin")
        plt.plot(x, model_cov, label="Model with cov", linestyle='--', lw=lw)
        # plt.fill_between(x, model_cov + 2 * model_cov_err, model_cov - 2 * model_cov_err, alpha=0.2, label=r"2$\sigma$ margin")

        plt.legend()

    # nuisance parameter profiling
    nuis = nuisance_profiling(D, Tcov, beta, uncorr)
    # nuis = nuisance_profiling(D, T, beta, uncorr)
    # nuis = 2*np.ones_like(nuis)*np.sign(nuis)
    # nuis[np.abs(nuis>3)] = 0
    D_shifted = D - np.sum(beta * nuis, axis=1)

    minchi2 = np.sum((D_shifted - T)**2 / uncorr**2) + np.sum(nuis**2)
    print(r"$\chi_{min}^2$/N")
    print(minchi2 / np.size(D))

    x1, D1, beta1, T1, Tcov1, D_sf1 = cut_by_Q2_and_sort_by_x(
        Q2, x, 3.5, D, beta, T, Tcov, D_shifted)[1:]
    x2, D2, beta2, T2, Tcov2, D_sf2 = cut_by_Q2_and_sort_by_x(
        Q2, x, 18, D, beta, T, Tcov, D_shifted)[1:]
    D1_sys = np.sqrt(np.sum(beta1**2, axis=1))
    D2_sys = np.sqrt(np.sum(beta2**2, axis=1))

    # plt.figure()
    D_er_sys = np.sqrt(np.sum(beta**2, axis=1))

    plot_data(x, D, D_er_sys, D_shifted,
              labels=[r"$\sigma_r$ with sys err", r"$\sigma_r$ shifted"],
              title="Data vs shifted")
    # plt.savefig("kuvat/agains data/Data vs shifted.png")

    title = "$Q^2$ = 3.5 $GeV^2$"
    plot_against_exp(x1, D1, D1_sys, D_sf1, T1, Tcov1, title)
    plt.savefig("Q2_3,5_lines.pdf")

    title = "$Q^2$ = 18 $GeV^2$"
    plot_against_exp(x2, D2, D2_sys, D_sf2, T2, Tcov2, title)
    plt.savefig("Q2_18_lines.pdf")

    plot_diff(D, D_shifted, D_er_sys)

    # plot_data(x, D, D_er_sys, T, labels=[r"$\sigma_r$ with sys err", r"model"], title="Data vs model")
    return D_shifted


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


def pick_samples(samples, N, par_limits=None):
    """
    Pick N random samples from the samples with some parameter limits (optional).

    Parameters
    ----------
    samples : ndarray
        The parameter samples to be cut.
    N : int
        How many samples are picked from samples.
    par_limits : list or ndarray, optional
        The limitations for parameters. Can be used to cut down samples to some exact region.
        The default is None.

    Returns
    -------
    ndarray
        N random samples.

    """
    if par_limits is not None: samples = cut_samples(samples, par_limits)
    # same as samples[np.random.randint(0, samples.shape[0], 100)] but without copies.
    return samples[np.random.choice(samples.shape[0], N)]


def samples_plot(samples, N, par_limits, emulator, x, Q2, y, sigma_r_exp, sigma_r_err, save=False):
    """
    Plot the emulator using N parameters from the posterior.

    Parameters
    ----------
    samples : array
        The parameter samples gotten from MCMC.
    N : int
        How many parameter samples are taken from samples.
    par_limits : array
        Parameters' limits
    emulator : function
        The emulator which generates predictions given a parametisation.
    x : array
        Bjorken x.
    sigma_r_exp : array
        Experimental sigma_r data.
    sigma_r_err : array
        The errors of experimental sigma_r data.
    save : bool, optional
        If hundred pars will be saved for same future plots. The default is False.

    Returns
    -------
    None.

    """
    # Lets only take data where srtq(s) = 318 GeV^2.
    rts = np.sqrt(Q2 / y / x)
    idx = np.where(np.isclose(rts, 318, rtol=0.01, atol=0.01))

    # N samples from posterior. Then calculated sigma_r with them, then average, then plotting.
    cut_samples(samples, par_limits)
    pars = pick_samples(samples, N)
    emu_sigma = emulator(pars)

    # If emulator returns standard deviations or covariance too
    if isinstance(emu_sigma, list) or isinstance(emu_sigma, tuple):
        emu_sigma = emu_sigma[0]
    elif np.ndim(emu_sigma) > 2: emu_sigma = emu_sigma[0]

    m = np.percentile(emu_sigma, [16, 50, 84], axis=0)
    # avg_sigma_r = np.mean(emu_sigma, axis=0)
    # std_sigma_r = np.std(emu_sigma, axis=0)
    avg_sigma_r = m[1]
    std_sigma_r = ((m[2] - m[0]) / 2)
    # print(np.max(100 * (avg_sigma_r - mens) / avg_sigma_r))
    # print(np.max(100 * (std_sigma_r - stdssdf) / std_sigma_r))

    plt.figure(figsize=(7, 5))
    #plt.title(str(N) + r' random samples from the cov posterior against HERA data with $\sqrt{s} = 318$ GeV$^2$.')

    def plot(x, sigma_r_exp, sigma_r_err, avg_sigma_r, std_sigma_r, Q2_val, temp=1):
        if temp == 0: plt.errorbar(x, sigma_r_exp, sigma_r_err, color='k', alpha=0.5, fmt='o',
                                   markersize=2, capsize=2, label="HERA data")
        else: plt.errorbar(x, sigma_r_exp, sigma_r_err, color='k', alpha=0.5, fmt='o',
                           markersize=2, capsize=2)
        plt.plot(x, avg_sigma_r, label=str(Q2_val) + " GeV$^2$")
        # label=r"$\sigma_r$ from posterior")
        plt.fill_between(x, avg_sigma_r + 2 * std_sigma_r,
                         avg_sigma_r - 2 * std_sigma_r, alpha=0.4)
        # label=r"2$\sigma$ margin")

    Q2vals = [45.0, 27.0, 15.0, 8.5, 4.5, 2.0]
    temp = 0
    for val in Q2vals:
        args = cut_by_Q2_and_sort_by_x(Q2[idx], x[idx], val, sigma_r_exp[idx], sigma_r_err[idx],
                                       avg_sigma_r[idx], std_sigma_r[idx])[1:]
        plot(*args, val, temp)
        temp += 1

    plt.xscale('log')
    plt.xlabel(r'$x_{Bj}$')
    plt.ylabel(r'$\sigma_r$')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.savefig('plot_hundpars.pdf')

    if save:
        np.savetxt("data/temp/hund_samps.dat", pars)

    plt.show()


def cut_by_Q2_and_sort_by_x(Q2, x, Q2_val, *other_vars_to_sort):
    Q2lim = Q2 == Q2_val
    Q2 = Q2[Q2lim]
    x = x[Q2lim]
    sort = np.argsort(x)
    Q2 = Q2[sort]
    x = x[sort]
    vars_ = []
    for var in other_vars_to_sort:
        vars_.append(var[Q2lim][sort])
    return Q2, x, *vars_


def cut(data):
    """
    Cut data with x<0.01 and 1<Q2<50.

    First column of data should be bjorken x and second Q2.

    Parameters
    ----------
    data : array
        Data to be cut. First column of data should be bjorken x and second Q2.

    Returns
    -------
    array
        Cut data according to limitations.

    """
    return data[(1 < data[:, 0]) & (data[:, 0] < 50) & (data[:, 1] < 0.01), :]


def load_light_data():
    """
    Load, cut and combine light quark data.

    Returns
    -------
    array
        The combined and cut light quark data.

    """
    #  charged current. Does not matter in our x, Q2 limits.
    # d1 = np.loadtxt(r"data/experiment data/ep/HERA1+2_CCem.dat", skiprows=1)
    # d2 = np.loadtxt(r"data/experiment data/ep/HERA1+2_CCep.dat", skiprows=1)
    # e- p. Not compatiple with calculated training data. Does not matter in our x, Q2 limits.
    # d3 = np.loadtxt(r"data/experiment data/ep/HERA1+2_NCem.dat", skiprows=1)
    d4 = np.loadtxt(r"data/experiment data/ep/HERA1+2_NCep_460.dat", skiprows=1)
    d5 = np.loadtxt(r"data/experiment data/ep/HERA1+2_NCep_575.dat", skiprows=1)
    d6 = np.loadtxt(r"data/experiment data/ep/HERA1+2_NCep_820.dat", skiprows=1)
    d7 = np.loadtxt(r"data/experiment data/ep/HERA1+2_NCep_920.dat", skiprows=1)
    light_data = np.vstack((d4, d5, d6, d7))
    # Constraints
    return cut(light_data)


def load_c_data():
    """
    Load and cuts the charm quark data.

    Returns
    -------
    array
        The cut charm quark data.

    """
    dc = np.loadtxt('data/experiment data/desy_charm.txt')
    return cut(dc)


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


def parameter_limits(n):
    """
    Get parameter limits.

    Parameters
    ----------
    n : int
        How many parameters are used in the setup.

    Returns
    -------
    array
        The parameter limits.

    """
    sigma0_limits = [10, 30]  # mb
    lambda_limits = [0.05, 0.5]
    Qs02_limits = [0.1, 5]
    mc_limits = [1, 2]  # GeV
    gamma_limits = [0.5, 2]
    if n == 5:
        return np.array([sigma0_limits, lambda_limits, Qs02_limits, gamma_limits, mc_limits])
    return np.array([sigma0_limits, lambda_limits, Qs02_limits, mc_limits])[:n]


def parameter_names(n):
    """
    Get parameters' names/labels.

    Parameters
    ----------
    n : int
        How many parameters are returned.

    Returns
    -------
    list
        The parameter names with units.

    """
    if n == 5:
        return [r'$\sigma_0$ [mb]', r'$\lambda$', '${Q^2}_{s0}$ [GeV$^2$]', r'$\gamma$', '$m_c$ [GeV]']
    return [r'$\sigma_0$ [mb]', r'$\lambda$', '${Q^2}_{s0}$ [GeV$^2$]', '$m_c$ [GeV]'][:n]
