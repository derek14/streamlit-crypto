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

def show_efficient_froniter(long_only=True):
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
  plt.tight_layout()
  plt.savefig("ef_scatter.png", dpi=200)
  plt.show()
  return ef.clean_weights()

def download_crypto_data(selected_pairs, start_date, interval = 86400):
  temp_dict = {}
  for pair in selected_pairs:
    temp_dict[pair] = HistoricalData(pair,interval,start_date, verbose=False).retrieve_data()["close"]
  crypto_df = pd.DataFrame.from_dict(temp_dict)
  crypto_df.dropna(axis='columns', inplace=True)
  return crypto_df