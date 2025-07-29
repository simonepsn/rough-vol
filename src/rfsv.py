import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import cholesky
from typing import List

def estimate_h_loglog_weighted(series: pd.Series, scales: List[int], q: int = 2):
    """
    Estimates the Hurst exponent (H) and volatility scale (nu) using a weighted
    log-log regression of absolute increments.

    Args:
        series (pd.Series): Time series data.
        scales (List[int]): List of time scales (tau) to use for increments.
        q (int): Order of the absolute moment (default is 2).

    Returns:
        Tuple[float, float, int]: Estimated H, nu, and the number of points used
                                  for regression (0 if not enough points).
    """
    log_tau, log_moments, weights = [], [], []
    for tau in scales:
        increments = series.diff(tau).dropna()
        n_obs = len(increments)
        if n_obs > 0:
            moment = np.mean(np.abs(increments) ** q)
            if moment > 0:
                log_tau.append(np.log(tau))
                log_moments.append(np.log(moment))
                weights.append(n_obs)
    
    if len(log_tau) < 2:
        return np.nan, np.nan, 0
    
    log_tau = np.array(log_tau)
    log_moments = np.array(log_moments)
    weights = np.array(weights)
    
    # Handle cases where all weights are zero or very small
    if np.sum(weights) == 0:
        return np.nan, np.nan, 0
            
    try:
        slope, intercept = np.polyfit(log_tau, log_moments, deg=1, w=weights)
        h = np.clip(slope / q, 0.01, 0.99) # Clip H to a reasonable range
        nu = np.sqrt(np.exp(intercept))
    except np.linalg.LinAlgError:
        # Handle cases where polyfit might fail (e.g., singular matrix)
        return np.nan, np.nan, 0
    
    return h, nu, len(log_tau)

# Function to build the FBM covariance matrix
def build_fbm_covariance_matrix(times: np.ndarray, h: float) -> np.ndarray:
    """
    Builds the covariance matrix for fractional Brownian motion (fBm).

    Args:
        times (np.ndarray): Array of time points.
        h (float): Hurst exponent.

    Returns:
        np.ndarray: The fBm covariance matrix.
    """
    n = len(times)
    t = times.reshape(-1, 1)
    abs_diff = np.abs(t - t.T)
    cov = 0.5 * (np.power(t, 2 * h) + np.power(t.T, 2 * h) - np.power(abs_diff, 2 * h))
    return cov

# Function for single-step forecasting
def forecast_single_step(
    past_log_rv: np.ndarray, 
    h: float, 
    nu: float, 
    horizon: int, 
    n_sims: int
) -> np.ndarray:
    """
    Performs a single step forecast using the RFSV model.

    Args:
        past_log_rv (np.ndarray): Array of past log realized volatility values.
        h (float): Hurst exponent.
        nu (float): Volatility scale.
        horizon (int): Number of steps ahead to forecast.
        n_sims (int): Number of Monte Carlo simulations (1 for conditional expectation).

    Returns:
        np.ndarray: Array of forecasted log realized volatility values.
    """
    n = len(past_log_rv)
    times = np.arange(1, n + horizon + 1) # Time points for observed and forecasted data
    cov = build_fbm_covariance_matrix(times, h)

    sigma_pp = cov[:n, :n] # Covariance of past data
    sigma_ff = cov[n:, n:] # Covariance of future data
    sigma_fp = cov[n:, :n] # Covariance between future and past data

    # Add a small regularization term to the diagonal of sigma_pp to prevent singularity
    sigma_pp_reg = sigma_pp + np.eye(n) * 1e-6 
    
    try:
        # Solve for the conditional mean
        solved = np.linalg.solve(sigma_pp_reg, past_log_rv / nu)
        mean = sigma_fp @ solved
    except np.linalg.LinAlgError as e:
        return np.full(horizon, np.nan)
    except ValueError as e: # Catch potential ValueError from solve if inputs are bad
        return np.full(horizon, np.nan)

    if n_sims == 1:
        # Return conditional expectation
        return (nu * mean).flatten()
    else:
        # Calculate conditional covariance for Monte Carlo simulations
        try:
            cov_cond = sigma_ff - sigma_fp @ np.linalg.solve(sigma_pp_reg, sigma_fp.T)
            # Ensure cov_cond is symmetric and positive semi-definite for Cholesky
            cov_cond = (cov_cond + cov_cond.T) / 2
            # Add a small epsilon to the diagonal for numerical stability before Cholesky
            cov_cond_reg = cov_cond + np.eye(horizon) * 1e-10
            
            L = cholesky(cov_cond_reg, lower=True)
            z = np.random.normal(size=(horizon, n_sims))
            sim_paths = mean[:, None] + L @ z
            return np.mean(nu * sim_paths.T, axis=0)
        except np.linalg.LinAlgError as e:
            return np.full(horizon, np.nan)


def rolling_forecast_rfsv(
    log_rv_series: pd.Series,
    scales: List[int],
    horizon: int,
    rolling_window: int = 252,
    n_sims: int = 1,
    freq: str = "D"
) -> pd.Series:
    """
    Computes RFSV forecasts of log-realized volatility for a given horizon.
    This function is designed to produce a single block of 'horizon' forecasts
    using the last 'rolling_window' observations from the input series.

    Args:
        log_rv_series (pd.Series): Time series of log realized variance (training data).
        scales (List[int]): Scales for H and nu estimation.
        horizon (int): Number of steps ahead to forecast.
        rolling_window (int): Length of past window to use for estimation.
        n_sims (int): Number of Monte Carlo paths (1 = conditional expectation).
        freq (str): Frequency of the time series (e.g., "D", "h", "5min").

    Returns:
        pd.Series: Forecasted log realized volatility values for the specified horizon.
                   The index will be a default integer index.
    """
    
    # Ensure rolling_window does not exceed the length of the series
    if rolling_window > len(log_rv_series):
        print(f"Warning: rolling_window ({rolling_window}) is greater than series length ({len(log_rv_series)}). Using full series length.")
        rolling_window = len(log_rv_series)

    # Use the last 'rolling_window' observations for estimation
    window = log_rv_series.iloc[-rolling_window:]
    
    h, nu, _ = estimate_h_loglog_weighted(window, scales)
    
    if np.isnan(h) or np.isnan(nu):
        forecast = np.full(horizon, np.nan)
    else:
        forecast = forecast_single_step(window.values, h, nu, horizon, n_sims)
    
    # Return the forecasts as a pandas Series with a default integer index
    return pd.Series(forecast)