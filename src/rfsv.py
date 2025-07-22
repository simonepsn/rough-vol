import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.linalg import cholesky
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error, mean_absolute_error



# THIS METHOD IS NOT WORKING FOR NOW, KEEPING IT JUST FOR FUTURE ANALYSIS
# ESTIMATES ARE ALWAYS (0.82-0.9) FOR H, ONLY WHEN UPPER BOUND IS HIGH ENOUGH 

def theoretical_acf(h, n_lags):
    """Computes theoretical ACF for an fBM-based process."""
    lags = np.arange(1, n_lags + 1)
    # Autocovariance for an incremental fBM
    autocov = 0.5 * (np.abs(lags - 1)**(2*h) - 2 * np.abs(lags)**(2*h) + np.abs(lags + 1)**(2*h))
    return autocov

def est_parameters(log_rv_series, max_lags=100):
    """
    Estimate Hurst exponent and extract vol-of-vol coefficient nu.

    Args:
        log_rv_series (pandas.Series): Log realized variance series.
        max_lags (int): Number of lags to use for ACF matching.

    Returns:
        tuple: (h_estimated, nu_estimated)
    """
    
    empirical_acf_vals = acf(log_rv_series, nlags=max_lags, fft=True)[1:]
    
    def objective_function(h):

        theoretical_acf_vals = theoretical_acf(h[0], max_lags)
        # "Loss function"
        return np.sum((empirical_acf_vals - theoretical_acf_vals)**2)

    # Minimize to find H
    result = minimize(objective_function, x0=[0.1], bounds=[(0.01, 0.49)])
    h_estimated = result.x[0]

    nu_estimated = np.std(log_rv_series)
    
    return h_estimated, nu_estimated

# Let's use log-log regression method instead


def estimate_h_loglog(series, scales, q=1):
    """
    Estimate Hurst exponent using log-log regression on moments.

    Args:
        series (pd.Series): Series.
        scales (list of int): Lag to analyze.
        q (int): Order of the moment to use (usually either 1 or 2).

    Returns:
        tuple: (H_est, R_squared)
    """
    moments = []
    for tau in scales:
        increments = series.diff(tau).dropna()
        moment = np.mean(np.abs(increments)**q)
        moments.append(moment)
    
    log_tau = np.log(scales)
    log_moments = np.log(moments)
    
    X = sm.add_constant(log_tau)
    y = log_moments
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    slope = results.params[1]
    
    h_estimated = slope / q
    
    return h_estimated, results.rsquared


# Now for the forecast we need to build the fBM var-cov matrix

def build_fbm_covariance_matrix(times, h):
    n = len(times)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            t_i = times[i]
            t_j = times[j]
            cov_matrix[i, j] = 0.5 * (
                np.power(t_i, 2 * h) + 
                np.power(t_j, 2 * h) - 
                np.power(np.abs(t_i - t_j), 2 * h)
            )
    return cov_matrix


def forecast_RFSV(past_log_rv_series, h, nu, horizon, truncation_window=252, n_sims=1):
    """
    Conditional forecasting optimised for RFSV using a truncated series (easier to invert).

    Args:
        past_log_rv_series (pd.Series): log_rv.
        h (float): Hurst exponent.
        nu (float): Vol-of-vol coefficient.
        horizon (int): Orizzonte di previsione.
        truncation_window (int): How many observations to consider before truncating.
        n_sims (int): Numbers of simulations paths to generate.

    Returns:
        pd.Series: Forecasted series.
    """
    if len(past_log_rv_series) > truncation_window:
        past_log_rv_series_truncated = past_log_rv_series.iloc[-truncation_window:]
    else:
        past_log_rv_series_truncated = past_log_rv_series

    n_past = len(past_log_rv_series_truncated)
    
    past_times = np.arange(1, n_past + 1)
    future_times = np.arange(n_past + 1, n_past + horizon + 1)
    all_times = np.concatenate([past_times, future_times])

    cov_global = build_fbm_covariance_matrix(all_times, h)
    
    sigma_pp = cov_global[:n_past, :n_past]
    sigma_ff = cov_global[n_past:, n_past:]
    sigma_fp = cov_global[n_past:, :n_past]
    sigma_pf = sigma_fp.T

    # np.linalg.solve is more stable and faster than np.linalg.inv
    past_W = past_log_rv_series_truncated.values / nu
    solved_part = np.linalg.solve(sigma_pp, past_W)
    
    mean_cond_W = sigma_fp @ solved_part
    cov_cond_W = sigma_ff - sigma_fp @ np.linalg.solve(sigma_pp, sigma_pf)

    last_known_date = past_log_rv_series.index[-1]
    freq = pd.infer_freq(past_log_rv_series.index)
    forecast_index = pd.date_range(start=last_known_date, periods=horizon + 1, freq=freq, inclusive='right')

    if n_sims > 1:
        L = cholesky(cov_cond_W + np.eye(horizon) * 1e-8, lower=True)
        z = np.random.normal(size=(horizon, n_sims))
        simulated_paths_W = mean_cond_W[:, np.newaxis] + L @ z
        simulated_paths_log_sigma = nu * simulated_paths_W.T

        mean_forecast = np.mean(simulated_paths_log_sigma, axis=0)
        return pd.Series(mean_forecast, index=forecast_index, name='rfsv_forecast_sim_mean')
    else:
        mean_forecast_log_sigma = nu * mean_cond_W
        return pd.Series(mean_forecast_log_sigma, index=forecast_index, name='rfsv_forecast')