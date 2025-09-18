import pandas as pd
import numpy as np
from scipy.stats import t
import logging
logging.basicConfig(level=logging.INFO)

class FactorsModelOLS:
    """
    Custom implementation of Ordinary Least Squares (OLS) regression
    for use in financial factor modeling and analysis.

    Attributes:
        _fit_intercept (bool): Whether to include an intercept in the model.
    """

    def __init__(self, fit_intercept=True):
        """
        Initializes the model.

        Args:
            fit_intercept (bool): Whether to include an intercept term. Default is True.
        """
        self._fit_intercept = fit_intercept

    def fit(self, factors, target):
        """
        Fits the OLS model to the given factors and target.

        Args:
            factors (np.ndarray or pd.DataFrame): Matrix of explanatory variables.
            target (np.ndarray or pd.Series): Vector of dependent variable values.

        Raises:
            ValueError: If the target is not one-dimensional.
        """
        # Save column names if DataFrame
        if isinstance(factors, pd.DataFrame):
            self._names = factors.columns

        # Convert inputs to NumPy arrays
        factorstype = type(factors)
        targettype = type(target)
        if not isinstance(factorstype, np.ndarray):
            factors = factors.to_numpy()
            logging.info(f'Factors data type: {factorstype} was automatically converted into np.ndarray')
        if not isinstance(targettype, np.ndarray):
            target = target.to_numpy()
            if len(target.shape) > 1:
                raise ValueError('Target must be one-dimensional')
            logging.info(f'Target data type: {targettype} was automatically converted into np.ndarray')

        self._x = factors
        if self._fit_intercept:
            self._x = np.column_stack((np.ones(self._x.shape[0]), self._x))
        self._y = target

        # Calculate beta coefficients
        self._betas = np.linalg.inv(self._x.T @ self._x) @ self._x.T @ self._y

        # Compute fitted values and residuals
        self._model_pred_values = self._x @ self._betas
        self._residuals = self._y - self._model_pred_values

        # Estimate residual variance
        n = len(self._x)
        p = len(self._betas)
        self._residuals_var = (self._residuals.T @ self._residuals) / (n - p)

        # Covariance matrix and standard errors
        self._betas_cov_matrix = self._residuals_var * np.linalg.inv(self._x.T @ self._x)
        self._betas_std = np.sqrt(np.diag(self._betas_cov_matrix))

        # t-statistics and p-values
        self._t = self._betas / self._betas_std
        self._pv = 2 * (1 - t.cdf(np.abs(self._t), df=n - p))

        logging.info('Model Fitted Successfully')

    def predict(self, features):
        """
        Predicts target values for given features.

        Args:
            features (np.ndarray or pd.DataFrame): New data to predict on.

        Returns:
            np.ndarray: Predicted target values.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_betas'):
            raise RuntimeError("Model was not fitted. Call 'fit' before prediction.")
        if not isinstance(features, np.ndarray):
            features = features.to_numpy()
        if self._fit_intercept:
            features = np.column_stack((np.ones(features.shape[0]), features))
        pred = features @ self._betas
        return pred

    def summary(self):
        """
        Returns a summary of the regression results.

        Returns:
            pd.DataFrame: A table with coefficients, standard errors, t-stats, and p-values.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_betas'):
            raise RuntimeError("Model is not fitted yet.")

        # Prepare feature names
        if hasattr(self, '_names'):
            names = list(self._names)
        else:
            names = [f'x{i}' for i in range(len(self._betas)) - int(self._fit_intercept)]
        if self._fit_intercept:
            names = ['Intercept'] + names

        df = pd.DataFrame({
            'Coefficient': self._betas,
            'Std': self._betas_std,
            't-Statistic': self._t,
            'p-Value': self._pv
        }, index=names)
        return df

    @property
    def factors_weights(self):
        """
        Returns the fitted regression coefficients.

        Returns:
            np.ndarray: Array of beta coefficients.

        Raises:
            KeyError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_betas'):
            raise KeyError("Can't find the model's weights. Fit the model first.")
        return self._betas

    @property
    def residuals(self):
        """
        Returns the model residuals.

        Returns:
            np.ndarray: Vector of residuals.

        Raises:
            KeyError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_residuals'):
            raise KeyError("Can't find the model's residuals. Fit the model first.")
        return self._residuals

    @property
    def p_values(self):
        """
        Returns the p-values of the estimated coefficients.

        Returns:
            np.ndarray: Array of p-values.

        Raises:
            KeyError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_pv'):
            raise KeyError("Can't find the model's p-values. Fit the model first.")
        return self._pv

    @property
    def t_statistics(self):
        """
        Returns the t-statistics of the estimated coefficients.

        Returns:
            np.ndarray: Array of t-statistics.

        Raises:
            KeyError: If the model has not been fitted yet.
        """
        if not hasattr(self, '_t'):
            raise KeyError("Can't find the model's t-statistics. Fit the model first.")
        return self._t