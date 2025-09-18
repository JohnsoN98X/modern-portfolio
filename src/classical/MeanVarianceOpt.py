import pandas as pd
import numpy as np
import cvxpy as cp
from .ModernPortfolio import ModernPortfolio
import logging
logging.basicConfig(level=logging.INFO)

class MeanVarianceOpt(ModernPortfolio):
    """
    A portfolio optimization class based on the mean-variance framework.

    Inherits from:
    --------------
    ModernPortfolio : Base class providing return, volatility, and covariance analytics
                      from log return data.

    Description:
    ------------
    Solves a mean-variance optimization problem to find the minimum-variance portfolio 
    that meets a target expected return, under user-defined constraints.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame of log returns, indexed by date and with asset tickers as columns.

    Attributes (populated after optimization)
    -----------------------------------------
    _best_weights : pd.Series
        Optimal portfolio weights per asset.

    _portfolio_log_returns : pd.Series
        Daily portfolio log returns based on optimal weights.

    _portfolio_returns : pd.Series
        Daily portfolio arithmetic returns.

    _portfolio_wealth : pd.Series
        Simulated cumulative portfolio value over time (starting from 1.0).

    _portfolio_return : float
        Expected annual return of the optimized portfolio.

    _optimal_variance : float
        Annualized portfolio variance.

    Methods
    -------
    optimize_weights(target_return, min_weight=None, max_weight=None, short_allowed=False)
        Solves the quadratic programming problem for given constraints.
    """

    def __init__(self, df):
        """
        Initialize the MeanVarianceOptimization object.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain log returns.
        """
        super().__init__(df)

    def optimize_weights(self, target_return, min_weight=None, max_weight=None, short_allowed=False):
        """
        Perform mean-variance optimization to find the optimal portfolio weights.

        Parameters
        ----------
        target_return : float
            The desired minimum annual expected return (in decimal form, e.g., 0.1 for 10%).

        min_weight : float, optional
            Minimum allowable weight for any single asset. If None, no lower bound is applied.

        max_weight : float, optional
            Maximum allowable weight for any single asset. If None, no upper bound is applied.

        short_allowed : bool, default=False
            If False, enforces non-negative weights (i.e., no short selling).

        Raises
        ------
        ValueError
            If target return is non-positive or if constraints are inconsistent.

        Notes
        -----
        - Covariance is based on daily log returns.
        - Returns and volatilities are annualized.
        - Weights sum to 1 (fully invested portfolio).
        """
        if target_return <= 0:
            raise ValueError('Target return must be a positive number')
        if not short_allowed and min_weight is not None and min_weight < 0:
            raise ValueError('Short selling is not allowed, but min_weight is negative.')
        if min_weight is not None and max_weight is not None and min_weight >= max_weight:
            raise ValueError('min_weight must be smaller than max_weight')

        cov_matrix = self._cov_matrix.values
        w = cp.Variable(len(self.df.columns))
        objective = cp.Minimize(w.T @ cov_matrix @ w)

        constraints = [
            cp.sum(w) == 1,
            w @ self._log_returns >= target_return
        ]
        if not short_allowed:
            constraints.append(w >= 0)
        if min_weight is not None:
            constraints.append(w >= min_weight)
        if max_weight is not None:
            constraints.append(w <= max_weight)

        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        if problem.status in ['optimal', 'optimal_inaccurate']:
            logging.info(f'Optimization completed successfully. Problem status: {problem.status}')
        else:
            logging.warning(f'Optimization failed. Status: {problem.status}')

        self._optimal_variance = problem.value
        self._best_weights = pd.Series(w.value, index=self.df.columns)

        log_history = self.df.mul(self._best_weights, axis=1).sum(axis=1)
        self._portfolio_log_returns = log_history
        self._portfolio_returns = np.exp(log_history) - 1
        self._portfolio_wealth = np.exp(log_history.cumsum())
        self._portfolio_return = np.exp(self._best_weights @ self._log_returns)

    @property
    def portfolio_variance(self):
        """Returns the variance of the optimized portfolio."""
        if not hasattr(self, '_optimal_variance'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return self._optimal_variance

    @property
    def best_weights(self):
        """Returns the optimal portfolio weights."""
        if not hasattr(self, '_best_weights'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return self._best_weights

    @property
    def portfolio_expected_return(self):
        """Returns the expected annual return of the optimized portfolio."""
        if not hasattr(self, '_portfolio_return'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return self._portfolio_return

    @property
    def portfolio_wealth(self):
        """Returns the cumulative wealth series of the optimized portfolio."""
        if not hasattr(self, '_portfolio_wealth'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return self._portfolio_wealth

    @property
    def portfolio_log_history(self):
        """
        Returns the daily log return history of the optimized portfolio.

        """
        if not hasattr(self, '_portfolio_log_returns'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return self._portfolio_log_returns

    @property
    def portfolio_history(self):
        """
        Returns the daily return history of the optimized portfolio.
    
        """
        if not hasattr(self, '_portfolio_log_returns'):
            raise AttributeError("No optimal value exists. Use 'optimize_weights' first")
        return np.exp(self._portfolio_log_returns)

    @property
    def summary(self):
        """Prints a formatted summary of the optimized portfolio statistics."""
        print('-' * 50)
        print('Portfolio Summary')
        print('-' * 50)
        print(f'Portfolio Variance: {self.portfolio_variance:.6f}')
        print(f'Portfolio Standard Deviation: {np.sqrt(self.portfolio_variance):.6f}')
        print(f'Expected Annual Return: {(self._portfolio_return - 1) * 100:.2f}%')