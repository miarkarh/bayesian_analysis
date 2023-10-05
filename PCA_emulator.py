# -*- coding: utf-8 -*-
"""
The Gaussian process emulator with PCA.

@author: Mikko Artturi Karhunen
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA_GPE:
    """
    Makes scaling and PCA transformations to a training data and trains a Gaussian process emulator
    for each first n principal component. Emulators' predictions and standard deviations or
    covariance matrices are inverse transformed and scaled back to the original data structure.
    I took an example from Chun Shen's implementation https://github.com/chunshen1987/bayesian_analysis/
    and from John Scott Moreland's implementation https://github.com/morelandjs/hic-param-est-qm18/
    for the emulator.

    One migth have to tune emulator's kernel for different setups.
    """

    def __init__(self, npc=2, whiten=True):
        """
        Initialize the class object.

        ----------
        n_pc : int, optional
            How many first n principal components are emulated. The default is 2.
        whiten : bool, optional
            Determines if the whitening is done in PCA. The default is True.

        Returns
        -------
        None.

        """
        self.pca = PCA(whiten=whiten, copy=False)
        self.scaler = StandardScaler(copy=False)
        self.whiten = whiten
        self.npc = npc

    def train(self, X, Y, par_limits, n_restarts=0, consta_k1_bounds=[1e-5, 1e5], length_scale_bounds=[0.01, 100], noise_level_bounds=[1e-10, 1e5]):
        """
        Scale and make PCA transformation to the training data and train the GP emulator for each
        first n principal components.

        ----------
        X : array
            The parameter vectors in an array of (n vectors, m parameters) which were used for
            calculating the training data.

        Y : array
            The generated training data from the model with the parameter vectors.
            Shape of (n parameters, m model or experimental points)

        par_limits : array
            The lower and upper limits of the parameters.

        n_restarts : int, optional
            The number of restarts of the optimizer in scikit learn's gaussian process regressor.
            Look scikit learn's GPR's documentation. The default is 0.

        consta_k1_bounds : pair of floats >= 0 or “fixed”, optional
            The lower and upper bound on constant_value. If set to “fixed”, constant_value cannot be
            changed during hyperparameter tuning. The default is [1e-5, 1e5].

        length_scale_bounds : pair of floats >= 0 or “fixed”, optional
            The lower and upper bound on ‘length_scale’. If set to “fixed”, ‘length_scale’ cannot be
            changed during hyperparameter tuning. The default is [0.01, 100] in my code.

        noise_level_bounds : pair of floats >= 0 or “fixed”, optional
            The lower and upper bound on ‘noise_level’. If set to “fixed”, ‘noise_level’ cannot be
            changed during hyperparameter tuning.. The default is [1e-10, 1e5].

        Returns
        -------
        None.

        """
        Ys = self.scaler.fit_transform(Y)
        Z = self.pca.fit_transform(Ys)[:, :self.npc]
        print(f'Explained variance of {self.npc} pc-components: '
              + str(np.sum(np.array(self.pca.explained_variance_ratio_)[:self.npc])))

        self.gps = []
        ptp = np.diff(par_limits).ravel()
        length_scale_bounds = np.outer(ptp, length_scale_bounds)
        # Matern if one wants to control smoothness of the aproximated function. (with nu parameter)
        # kernel = C(1., consta_k1_bounds) * Matern(ptp, length_scale_bounds, nu=1.5)
        kernel = C(1., consta_k1_bounds) * RBF(ptp, length_scale_bounds) + WhiteKernel(0.01, noise_level_bounds)
        for z in Z.T:
            gp = GPR(kernel=kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=n_restarts,
                     copy_X_train=False)
            gp.fit(X, z)
            self.gps.append(gp)

        self._trans_matrix = (
            self.pca.components_  # V "right singular vectors"
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])  # whitening?
            * self.scaler.scale_  # Scaling by standard deviations
        )
        # Compute the partial transformation for the first n principal components that are actually
        # emulated.
        A = self._trans_matrix[:self.npc]
        self._var_trans = np.einsum('ki,kj->kij', A, A, optimize=False).reshape(self.npc, self.pca.n_features_in_**2)

        # Compute the covariance matrix for the remaining neglected PCs (truncation error).
        # These components always have variance == 1.
        B = self._trans_matrix[self.npc:]
        self._cov_trunc = np.dot(B.T, B)

        # This is to add small term to diagonal for numerical stability, but I don't add.
        self._cov_trunc.flat[::self.pca.n_features_in_ + 1] += 0  # 1e-8 * self.scaler.var_

    def _inverse_transform(self, Z):
        """
        Inverse transform principal components to original data structure.

        Parameters
        ----------
        Z : array
            The array in PCA space.

        Returns
        -------
        Y : array
            The transformed array to original space.

        """
        Y = np.dot(Z, self._trans_matrix[:Z.shape[-1]])
        Y += self.scaler.mean_
        return Y

    def predict(self, X, return_std=False, return_cov=False, extra_std=0):
        """
        Takes parameter vector sample or samples and returns the emulator's predictions means and
        standard deviations or covariances at each data point. Also makes inverse pca
        transformations and scaling for emulators outputs back to original structure.

        Parameters
        ----------
        X : array
            The parameter samples to emulate with. Shape of (n samples, m dim) or only (m dim) if only one
            parameter sample is given.
        return_std : array, optional
            If emulator's predictions' standard deviations are to be returned too.
            The default is False.
        return_cov : array, optional
            If emulator's covariance is to be returned too. The default is False.
        extra_std : int, float, or array, optional
            Additional uncertainty which is added to each GP's predictive uncertainty.
            It may either be a scalar or an array-like of length n samples. The default is 0.

        Returns
        -------
        array, array
            Returns the means and standard deviations or covariances of the emulators predictions
            for each data point. Shape is (n samples, m data points) for means and standard
            deviations each, and (n samples, m datapoints, m datapoints) for covariance matrix

        """
        # Step for compapility with scikit-learn gpr
        dimX = np.ndim(X)
        if dimX == 1: X = np.array([X])

        # Returns means and covariance matrices for each npc
        gp_mean = [gp.predict(X, return_cov=(return_cov or return_std)) for gp in self.gps]
        if (return_cov or return_std):
            gp_mean, gp_cov = zip(*gp_mean)

        gp_means = np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        Y = self._inverse_transform(gp_means)
        if dimX == 1: Y = Y[0]
        if return_std or return_cov:
            # The covariance matrices have only diagonal variances in pca space.
            gp_var = np.concatenate([c.diagonal()[:, np.newaxis] for c in gp_cov], axis=1)

            # Add extra uncertainty to predictive variance.
            extra_std = np.array(extra_std, copy=False).reshape(-1, 1)
            gp_var += extra_std**2

            # Compute the covariance at each sample point using the pre-calculated arrays.
            cov = np.dot(gp_var, self._var_trans).reshape(X.shape[0], self.pca.n_features_in_, self.pca.n_features_in_)
            if self.whiten: cov += self._cov_trunc

            if return_cov:
                if dimX == 1: cov = cov[0]
                return Y, cov

            if return_std:
                n = cov.shape[0]
                # The diagonals should be larger than zero, but abs is just in case.
                std = np.sqrt(np.abs([np.diagonal(cov[i]) for i in range(n)]))
                if dimX == 1: std = std[0]
                return Y, std
        return Y

    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at X.

        Parameters
        ----------
        X : array
            The parameter samples to emulate and sample the emulator with.
        n_samples : int, optional
            Number of samples drawn from the Gaussian process per data point. The default is 1.
        random_state : int, optional
            Determines random number generation to randomly draw samples. Pass an int for
            reproducible results across multiple function calls. The default is None.

        Returns
        -------
        Y : array
            The prediction samples from the emulator transformed to original space and data
            structure. Shape of (n parameter samples, m emulator samples, k datapoints).

        """
        # Step for compapility with scikit-learn gpr
        if np.ndim(X) == 1: X = np.array([X])

        gp_samples = [gp.sample_y(X, n_samples=n_samples, random_state=random_state)[:, :, np.newaxis] for gp in self.gps]
        gp_samples = np.concatenate(gp_samples + [np.random.standard_normal((X.shape[0], n_samples, self.pca.n_components_ - self.npc))], axis=2)

        Y = self._inverse_transform(gp_samples)
        return Y
