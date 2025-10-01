import numpy as np
from sklearn.covariance import LedoitWolf
import warnings

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

    
    def fit(self, allocation='minvar', cov_method='ledoit', normalize=True, risk_target=None):
        """
        Fit the PCA risk allocation model and compute portfolio weights.

        Parameters
        ----------
        allocation : str, default='minvar'
            Method for distributing risk across principal components. Options:
            - 'minvar' : All risk on the last principal component (lowest variance).
            - 'equal' : Equal risk allocation to all components.
            - 'proportional' : Risk proportional to each eigenvalue (variance explained).
            - 'inverse' : Risk inversely proportional to each eigenvalue.
        cov_method : str, default='ledoit'
            Covariance estimation method. Options:
            - 'ledoit' : Ledoit-Wolf shrinkage estimator.
            - 'sample' : Sample covariance matrix.
        normalize : bool, default=True
            If True, weights are normalized so that their sum equals 1.
            In this case, risk_target is ignored.
            If False, weights are scaled so that the portfolio volatility equals risk_target.
        risk_target : float, optional
            Target portfolio volatility. Must be provided if normalize=False.
        """
        # --- covariance estimation ---
        if cov_method == 'ledoit':
            lw = LedoitWolf().fit(self._returns)
            self._cov = lw.covariance_
        elif cov_method == 'sample':
            self._cov = np.cov(self._returns.T)
    
        # --- input validation ---
        if normalize:
            if risk_target is not None:
                warnings.warn("normalize=True; risk_target will be ignored")
        else:
            if risk_target is None:
                raise ValueError("normalize=False requires risk_target")
    
        # --- eigen decomposition ---
        evalues, evectors = np.linalg.eigh(self._cov)
        index = evalues.argsort()[::-1]
        self._evalues = evalues[index]
        self._evectors = evectors[:, index]
    
        # --- risk distribution ---
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
    
        # --- compute loadings ---
        # If normalize=True, risk_target is ignored (scale=1)
        scale = risk_target if (risk_target is not None and not normalize) else 1.0
        self._loads = scale * (self._risk_dist / self._evalues) ** 0.5
    
        # --- compute raw weights ---
        w = self._evectors @ self._loads.reshape(-1, 1)
        w = w.ravel()
        self._sigma = float(np.sqrt(w @ self._cov @ w))
    
        # --- normalize or scale ---
        if normalize:
            denom = np.sum(w)
            if denom == 0:
                raise ValueError("all-zero weight vector; cannot normalize.")
            w = w / denom
        else:
            w = w * (risk_target / self._sigma)
    
        self._w = w


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