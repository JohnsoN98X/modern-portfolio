import numpy as np
from sklearn.covariance import LedoitWolf

class PCARiskAllocator:
    """
    PCA-based risk allocation portfolio optimizer.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset return series (log or raw prices depending on `logs` flag).
    logs : bool, default=True
        Whether the input is in log returns (True) or raw reruns (False).
    """
    
    def __init__(self, returns, logs=True):
        if not isinstance(returns, np.ndarray):
            raise ValueError('returns must be NumPy ndarray')
        self._returns = returns if logs else np.log(returns)

    def __repr__(self):
        """
        Return a string summary of the PCA portfolio optimizer.
        """
        cls_name = self.__class__.__name__
        n_assets = self._returns.shape[1] if self._returns.ndim > 1 else 1
        parts = [f"{cls_name}(assets={n_assets})"]
        parts.append(
            f"PCA: sigma={self._sigma:.4f}" if hasattr(self, '_w') else 'PCA: not fitted'
        )
        return ' | '.join(parts)

    def fit(self, risk_target, allocation='minvar', cov_method='ledoit'):
        """
        Fit the PCA risk allocation model and compute portfolio weights.

        Parameters
        ----------
        risk_target : float
            Target portfolio volatility.
        allocation : str, default='minvar'
            Method for distributing risk across principal components. Options:
            - 'minvar' : All risk on the last principal component (lowest variance).
            - 'equal' : Equal risk allocation to all components.
            - 'proportional' : Risk proportional to each eigenvalue (variance explained).
            - 'inverse' : Risk inversely proportional to each eigenvalue.
        """
        if cov_method == 'ledoit':
            lw = LedoitWolf().fit(self._returns)
            self._cov = lw.covariance_
        elif cov_method == 'sample':
            self._cov = np.cov(self._returns.T)

        evalues, evectors = np.linalg.eigh(self._cov)
        index = evalues.argsort()[::-1]
        self._evalues = evalues[index]
        self._evectors = evectors[:, index]

        if allocation == 'minvar':
            self._risk_dist = np.zeros(len(self._evalues))
            self._risk_dist[-1] = 1
        elif allocation == 'equal':
            self._risk_dist = np.ones(len(self._evalues)) / len(self._evalues)
        elif allocation == 'proportional':
            self._risk_dist = self._evalues / self._evalues.sum()
        elif allocation == 'inverse':
            inv = 1 / self._evalues
            self._risk_dist = inv / inv.sum()
        else:
            raise KeyError(f"Unexpected allocation: {allocation}")

        self._loads = risk_target * (self._risk_dist / self._evalues)**0.5
        w = self._evectors @ self._loads.reshape(-1, 1)
        w = w.ravel()
        self._sigma = float(np.sqrt(w @ self._cov @ w))
        self._w = w * (risk_target / self._sigma)

    @property
    def weights(self):
        """
        Get the PCA-allocated portfolio weights.

        Returns
        -------
        np.ndarray
            Vector of asset weights.
        """
        if not hasattr(self, '_w'):
            raise KeyError('no pca weights were found')
        return self._w

    @property
    def sigma(self):
        """
        Get the realized portfolio volatility under PCA allocation.

        Returns
        -------
        float
            Portfolio standard deviation.
        """
        if not hasattr(self, '_sigma'):
            raise ValueError('no pca sigma was found')
        return self._sigma

    @property
    def evectors(self):
        """
        Return the principal component directions (eigenvectors).

        Returns
        -------
        np.ndarray
            Matrix of eigenvectors.
        """
        if not hasattr(self, '_evectors'):
            raise KeyError('no eigencevtors were found')
        return self._evectors

    @property
    def evalues(self):
        """
        Return the eigenvalues (variances explained by each principal component).

        Returns
        -------
        np.ndarray
            Vector of eigenvalues.
        """
        if not hasattr(self, '_evalues'):
            raise KeyError('no eigenvalues were found')
        return self._evalues

    @property
    def loadings(self):
        """
        Get the loadings (risk budget allocation) applied to each component.

        Returns
        -------
        np.ndarray
            Vector of loadings.
        """
        if not hasattr(self, '_loads'):
            raise KeyError('no loadings were found')
        return self._loads

    @property
    def betas(self):
        """
        Return the exposure (beta) of the PCA portfolio to each principal component.

        Returns
        -------
        np.ndarray
            Vector of betas.
        """
        betas = self._evectors.T @ self._w
        return betas

    @property
    def risk_contribution(self):
        """
        Compute the relative risk contribution of each principal component to total variance.

        Returns
        -------
        np.ndarray
            Vector of risk contributions summing to 1.
        """
        betas = self.betas
        rc = (betas**2) * self._evalues
        return rc / rc.sum()

    @property
    def explained_variance_ratio(self):
        """
        Return the proportion of total variance explained by each principal component.

        Returns
        -------
        np.ndarray
            Vector of explained variance ratios (summing to 1).
        """
        return self._evalues / self._evalues.sum()

    @property
    def portfolio(self):
        """
        Compute the time series of the PCA-weighted portfolio.

        Returns
        -------
        np.ndarray
            Portfolio returns as a 1D array.
        """
        return self._returns @ self._w