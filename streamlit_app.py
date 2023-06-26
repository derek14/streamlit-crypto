from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from utils import Cryptocurrencies, download_crypto_data
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
"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

crypto_pairs = Cryptocurrencies().find_crypto_pairs()
valid_crypto_pairs = crypto_pairs[(crypto_pairs["fx_stablecoin"]==False)&(crypto_pairs["status"]=="online")&(crypto_pairs["display_name"].str.endswith("/USD"))]

st.title("Crypto Portfolio Optimizer")

st.header("Inputs")
selected_pairs = st.multiselect(
    'What crypto pairs do you want to include?',
    valid_crypto_pairs["id"].unique().tolist(),
    ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOGE-USD', 'SOL-USD', 'LTC-USD'])

start_date = st.date_input(
    "Start date of data analysis",
    datetime.date(2022, 6, 25))

crypto_df = download_crypto_data(selected_pairs, start_date)

st.header("Risk Modelling")
mu = expected_returns.ema_historical_return(crypto_df, frequency=365)
Sigma = risk_models.exp_cov(crypto_df, frequency=365)
st.subheader("Correlation Plot")
st.text("A correlation plot is a graphical representation of the correlation between two or more variables. It is a scatter plot with the values of one variable on the x-axis and the values of the other variable on the y-axis. Each point on the plot represents a pair of values for the two variables. The position of the point on the plot indicates the values of the two variables, and the color or size of the point indicates the strength of the correlation between the variables. The correlation between two variables is a measure of the strength and direction of the relationship between them. A positive correlation means that as one variable increases, the other variable also tends to increase. A negative correlation means that as one variable increases, the other variable tends to decrease. A correlation of zero means that there is no relationship between the variables.")
# st.pyplot(plotting.plot_covariance(risk_models.exp_cov(crypto_df, frequency=365), plot_correlation=True))