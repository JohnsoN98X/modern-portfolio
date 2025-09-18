import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)
from .ModernPortfolio import ModernPortfolio

class CAPM(ModernPortfolio):
    """
    Capital Asset Pricing Model (CAPM) analysis based on log returns.

    Inherits From
    --------------
    ModernPortfolio : Base class providing annualized returns, volatilities, 
                      covariance, and correlation matrices from log return data.

    Description
    ------------
    This class extends the ModernPortfolio framework by incorporating CAPM-based 
    metrics such as beta, Sharpe ratio, and Treynor ratio, and visualizing the 
    Capital Market Line (CML).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing **log returns** of individual assets, indexed by dates.

    r_free : float
        Annualized risk-free rate, expressed in decimal (e.g., 0.03 for 3%).

    market_data : pd.Series or np.ndarray
        Daily log returns of the market index (e.g., S&P 500), indexed similarly to `df`.

    Attributes
    ----------
    market_data : pd.Series or np.ndarray
        Daily log returns of the market portfolio.

    Rm : float
        Annualized return of the market portfolio.

    Rf : float
        Annualized risk-free rate.

    _returns : pd.Series
        Annualized arithmetic returns of each asset.

    _market_std : float
        Annualized standard deviation of the market.

    _port_std : float
        Average annualized standard deviation across portfolio assets.

    _cml_slope : float
        Slope of the Capital Market Line, calculated as (Rm - Rf) / σ_market.

    Methods
    -------
    get_cml_curve(max_vol=1, points=1000)
        Generates the Capital Market Line (CML) curve data.

    plot_cml_curve(title=None, figsize=(10,4), text=True, legend=True)
        Plots the Capital Market Line along with the position of the assets in risk-return space.

    ratios
        Computes and returns a DataFrame with Beta, Sharpe ratio, and Treynor ratio for each asset.
    """

    def __init__(self, df, r_free, market_data):
        """
        Initialize the CAPM analysis class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of log returns of individual assets.

        r_free : float
            Annualized risk-free rate.

        market_data : pd.Series or np.ndarray
            Daily log returns of the market portfolio.
        """
        super().__init__(df)

        if not isinstance(market_data, (pd.Series, pd.DataFrame)):
            raise ValueError("Market data must be a pandas Series or pandas DataFrame")
            
        if not isinstance(market_data.index, pd.DatetimeIndex):
            indextype = type(market_data.index)
            market_data.index = pd.to_datetime(market_data.index)
            logging.info(f"market_data index type: {indextype} was automatically converted into pd.DatetimeIndex.")
            
        self.market_data = market_data
        self.Rm = self._calculate_annual_returns(market_data)
        self.Rf = r_free
        self._returns = np.exp(self._log_returns)
        self._market_std = market_data.std(ddof=1) * np.sqrt(252)
        self._port_std = np.mean(self._volatilities)
        self._cml_slope = (self.Rm - self.Rf) / self._market_std

    def _cml_return(self, volatility):
        """
        Calculate the expected return on the Capital Market Line (CML)
        for a given level of portfolio volatility.
    
        Parameters
        ----------
        volatility : float or np.ndarray
            Annualized portfolio volatility (standard deviation).
            Can be a single float value or an array of values.
    
        Returns
        -------
        float or np.ndarray
            Expected return(s) according to the CML formula:
            E[R_p] = R_f + [(R_m - R_f) / σ_m] * σ_p
        """
        return self.Rf + self._cml_slope * volatility

    def get_cml_curve(self, max_vol=1, points=1000):
        """
        Generate points on the Capital Market Line.

        Parameters
        ----------
        max_vol : float, optional (default=1)
            Maximum x-axis value (volatility) for plotting the CML.

        points : int, optional (default=1000)
            Number of interpolation points on the CML curve.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['Volatility', 'Expected Return'] representing the CML.
        """
        sigmas = np.linspace(0, max_vol, points)
        returns = self._cml_return(sigmas)
        return pd.DataFrame({'Volatility': sigmas, 'Expected Return': returns})

    def plot_cml_curve(self, title=None, figsize=(10, 4), text=True, legend=True):
        """
        Plot the Capital Market Line with asset points overlaid.

        Parameters
        ----------
        title : str, optional
            Title for the plot.

        figsize : tuple, optional (default=(10,4))
            Size of the figure.

        text : bool, default=True
            Whether to annotate assets with their tickers.

        legend : bool, default=True
            Whether to show the legend.
        """
        plt.figure(figsize=figsize, dpi=150)
        returns = self.get_cml_curve()
        plt.plot(returns['Volatility'], returns['Expected Return'], label='CML',
                 color='#4C8BF5', linestyle='--')

        plt.scatter(
            x=self._volatilities,
            y=self._returns - 1,
            alpha=0.7, s=4,
            c=self._returns / self._volatilities
        )

        plt.xlabel('Volatility', font='david')
        plt.ylabel('Expected Return', font='david')
        plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        plt.xticks(font='david')
        plt.yticks(font='david')

        if text:
            for asset in self.df.columns:
                x = self._volatilities[asset]
                y = np.exp(self._log_returns[asset]) - 1
                plt.text(x, y, asset, fontsize=8)

        if title:
            plt.title(title, font='david', size=14, weight='bold')

        plt.grid(ls='--')
        if legend:
            plt.legend(edgecolor='k', prop={'family': 'david'})
        sns.despine()
        plt.tight_layout()

    @property
    def ratios(self):
        """
        Return a DataFrame of CAPM-related performance metrics per asset.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
            - 'beta'    : CAPM beta of each asset (cov(asset, market) / var(market))
            - 'sharpe'  : Sharpe ratio of each asset
            - 'treynor' : Treynor ratio of each asset
        """
        market_var = self.market_data.var(ddof=1)

        # Beta per asset
        betas = []
        for col in self.df.columns:
            cov = self.df[col].cov(self.market_data)
            betas.append(cov / market_var)

        # Sharpe ratio
        sharps = (self._returns - self.Rf) / self._volatilities

        # Treynor ratio
        treynors = (self._returns - self.Rf) / np.array(betas)

        return pd.DataFrame(index=self.df.columns,
                            data={
                                'beta': betas,
                                'sharpe': sharps,
                                'treynor': treynors
                            })
