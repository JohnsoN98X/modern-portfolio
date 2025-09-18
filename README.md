# 📊 Portfolio Optimization – Classical & Modern Approaches

This project provides a practical implementation of **Portfolio Optimization**, combining both **classical theories** (e.g., Markowitz, CAPM) and **modern approaches** (e.g., PCA-based risk allocation).  
The goal is to bridge theoretical rigor with numerical tools that are applicable in real-world portfolio construction and financial analysis.

> ⚠️ **Note:** This project does **not** aim to present any scientific innovation or empirical analysis of portfolio results.  
> Its sole purpose is to offer a practical, modular framework for applying both classical and modern portfolio optimization methods, along with examples of how to use them effectively.

---

## 📦 Project Structure

The project is divided into two main modules:

### 🔹 Classical Methods
Located under `src/classical` and `notebooks/classical`.

1. **`MeanVarianceOpt`**  
   Performs **Mean-Variance Optimization** as defined in modern finance literature.  
   While most academic references focus on analytical solutions, these are often impractical.  
   This class performs **convex numerical optimization** using the `cvxpy` library.

2. **`CAPM`**  
   A lightweight, high-level utility class for quick analysis based on the **Capital Asset Pricing Model (CAPM)**.  
   It provides easy access to key financial ratios (Beta, Sharpe, Treynor) and generates intuitive visualizations to compare asset performance.

3. **`ModernPortfolio`**  
   An **abstract base class** designed to encapsulate shared attributes and methods.  
   This class is **not intended to be used directly**, but serves as the foundation for the two classes above.

---

### 🔹 Modern Methods
Located under `src/modern` and `notebooks/modern`.

1. **`PCARiskAllocator`**  
   Implements **risk allocation based on Principal Component Analysis (PCA)**.  
   Instead of allocating directly to assets, risk is distributed across principal components according to a chosen scheme (`minvar`, `equal`, `proportional`, `inverse`).  
   The portfolio is then reconstructed from these components and scaled to a target volatility.  
   This allows exploration of risk-based diversification beyond traditional variance minimization.

---

## 🚀 Getting Started
### 🧰 Requirements

| Purpose               | Libraries Needed                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| Running the classes   | `pandas`, `numpy`, `cvxpy` *(for MeanVarianceOpt only)*, `scikit-learn` *(for PCA utilities)* |
| Running the notebooks | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`              |

---


## 📡 Data Source & Usage

The dataset used for testing was retrieved using the `ib_insync` library connected to Interactive Brokers (IB),  
and also via the `yfinance` API for reproducibility.  
Due to IB’s licensing policy, **raw market data is excluded** from this repository.

---

## ⚠️ Disclaimer

All tools and methods in this project are provided **strictly for educational and research purposes**.  
They do not constitute investment advice or financial recommendations.  
The author assumes no liability for any actions taken based on this code or its output.