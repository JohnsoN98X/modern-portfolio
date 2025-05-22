import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class ModernPortfolio:
    """
    A foundational class for modern portfolio analysis, based on log returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing log returns of financial assets.
        - Columns represent individual assets (e.g., tickers).
        - Index must be a pandas DateTimeIndex (will be auto-converted if needed).
        - Missing values (NaNs) are allowed but a warning will be issued.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame of log returns (after DateTimeIndex validation).
    
    _log_returns : pd.Series
        The annualized log returns for each asset.
    
    _volatilities : pd.Series
        The annualized volatility (standard deviation) of log returns.
    
    _cov_matrix : pd.DataFrame
        The covariance matrix of daily log returns.
    
    _correlations : pd.DataFrame
        The correlation matrix of daily log returns.

    Notes
    -----
    - Input data must represent **logarithmic returns**, not price levels or arithmetic returns.
    - Annualized returns are computed as the total log return divided by the number of years in the data range.
    - Volatilities are scaled using √252 to reflect trading days in a year.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        if df.isna().values.any():
            logging.warning("Input DataFrame contains NaN values.")

        if not isinstance(df.index, pd.DatetimeIndex):
            indextype = type(df.index).__name__
            df.index = pd.to_datetime(df.index)
            logging.info(f"DataFrame index type: {indextype} was automatically converted into pd.DatetimeIndex.")
            print(f'index type: {type(df.index)}')

        self.df = df
        self._log_returns = self._calculate_annual_returns(df)
        self._volatilities = df.std(ddof=1) * np.sqrt(252)
        self._cov_matrix = df.cov()
        self._correlations = df.corr()

    def _calculate_annual_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates annualized log returns for each asset.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of log returns.

        Returns
        -------
        pd.Series
            Annualized log returns.
        """
        logs_sum = df.sum()
        date_range = (df.index[-1] - df.index[0]).days
        num_years = date_range / 365.25
        log_returns = logs_sum / num_years
        return log_returns

    

    @property
    def volatilities(self) -> pd.Series:
        """
        Returns annualized volatility (standard deviation) of each asset.
        
        Returns
        -------
        pd.Series
            Annualized volatilities for each asset.
        """
        if not hasattr(self, '_volatilities'):
            raise AttributeError("Volatilities have not been calculated.")
        return self._volatilities

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """
        Returns the covariance matrix of log returns (not annualized).
        
        Returns
        -------
        pd.DataFrame
            Daily covariance matrix of log returns.
        """
        if not hasattr(self, '_cov_matrix'):
            raise AttributeError("Covariance matrix is not available.")
        return self._cov_matrix

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Returns the correlation matrix of log returns.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix of daily log returns.
        """
        if not hasattr(self, '_correlations'):
            raise AttributeError("Correlation matrix is not available.")
        return self._correlations

