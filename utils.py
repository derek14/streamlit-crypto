import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from pypfopt import risk_models, plotting
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from Historic_Crypto import Cryptocurrencies, HistoricalData, LiveCryptoData
import numpy as np
from datetime import datetime
import streamlit as st

@st.cache_data
def convert_df_to_csv(df):
    csv = df.to_csv(index=True).encode('utf-8')
    return csv

@st.cache_data
def download_valid_crypto_list():
  crypto_pairs = Cryptocurrencies().find_crypto_pairs()
  valid_crypto_pairs = crypto_pairs[(crypto_pairs["fx_stablecoin"]==False)&(crypto_pairs["status"]=="online")&(crypto_pairs["display_name"].str.endswith("/USD"))]
  return valid_crypto_pairs["id"].unique().tolist()

def download_crypto_data(selected_pairs, start_date, interval = 86400):
  temp_dict = {}
  for pair in selected_pairs:
    temp_dict[pair] = download_single_crypto_data(pair, start_date, interval)["close"]
  crypto_df = pd.DataFrame.from_dict(temp_dict)
  crypto_df.dropna(axis='columns', inplace=True)
  return crypto_df

@st.cache_data
def download_single_crypto_data(pair, start_date, interval = 86400):
   return HistoricalData(pair,interval,start_date, verbose=False).retrieve_data()
   
def ordered_dict_to_dataframe(ordered_dict):
    df = pd.DataFrame(list(ordered_dict.items()), columns=['TICKER', 'WEIGHT'])
    df = df.set_index("TICKER")
    return df

def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    plotting._plot_io(**kwargs)

    return fig

def show_efficient_froniter(mu, Sigma, long_only=True, **kwargs):
  ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1) if long_only else (None, None))
  fig, ax = plt.subplots()
  ef_max_sharpe = ef.deepcopy()
  plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

  # Find the tangency portfolio
  ef_max_sharpe.max_sharpe()
  ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
  ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

  # Generate random portfolios
  n_samples = 10000
  w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
  rets = w.dot(ef.expected_returns)
  stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
  sharpes = rets / stds
  ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

  # Output
  ax.set_title("Efficient Frontier with random portfolios")
  ax.legend()
  plotting._plot_io(**kwargs)
  return ef.clean_weights(), fig

