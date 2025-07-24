import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.linalg import cholesky
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def forecast_RFSV(past_log_rv_series, h, nu, horizon, freq, truncation_window=252, n_sims=1, use_last_value=True, index=None):
    """
    RFSV forecast starting from the last observed value.
    
    Args:
        past_log_rv_series (pd.Series): Series.
        h (int): Estimated Hurst parameter.
        nu (int): Estimated vol-of-vol parameter.
        horizon (int): Forecast horizon.
        freq (str): Frequency of the forecast.
        truncation_window (int): Number of past observations to consider.
        n_sims (int): Number of simulations to run.
        use_last_value (bool): Whether to use the last observed value to shift the forecast.
        index (pd.DatetimeIndex): Index for the forecast series.

    Returns:
        forecast_series (pd.Series): Forecasted series.

    """
    # Debug: Check input data quality
    print(f"RFSV Debug - Input range: {past_log_rv_series.min():.3f} to {past_log_rv_series.max():.3f}")
    print(f"RFSV Debug - H: {h:.3f}, nu: {nu:.3f}, horizon: {horizon}")
    
    if len(past_log_rv_series) > truncation_window:
        past_log_rv_series_truncated = past_log_rv_series.iloc[-truncation_window:]
    else:
        past_log_rv_series_truncated = past_log_rv_series

    n_past = len(past_log_rv_series_truncated)
    
    # Use more robust time scaling for high-frequency data
    if freq in ['5T', '5min']:
        time_scale = 1.0 / (288 * 252)  # Scale to annual units
    elif freq in ['h', 'H']:
        time_scale = 1.0 / (24 * 252)   # Scale to annual units  
    else:
        time_scale = 1.0 / 252           # Daily scale
    
    past_times = np.arange(1, n_past + 1) * time_scale
    future_times = np.arange(n_past + 1, n_past + horizon + 1) * time_scale
    all_times = np.concatenate([past_times, future_times])

    try:
        cov_global = build_fbm_covariance_matrix(all_times, h)
        
        # Add regularization for numerical stability
        regularization = 1e-6 * np.eye(len(all_times))
        cov_global += regularization
        
        sigma_pp = cov_global[:n_past, :n_past]
        sigma_ff = cov_global[n_past:, n_past:]
        sigma_fp = cov_global[n_past:, :n_past]
        sigma_pf = sigma_fp.T

        past_W = past_log_rv_series_truncated.values / nu
        
        # Use more stable solver
        try:
            solved_part = np.linalg.solve(sigma_pp, past_W)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            solved_part = np.linalg.pinv(sigma_pp) @ past_W
        
        mean_cond_W = sigma_fp @ solved_part
        cov_cond_W = sigma_ff - sigma_fp @ np.linalg.pinv(sigma_pp) @ sigma_pf

        if n_sims > 1:
            # Enhanced conditional covariance for better variability preservation
            if freq in ['5T', '5min']:
                variability_boost = 2.0  # Increase variability for 5-min data
            elif freq in ['h', 'H']:
                variability_boost = 1.5  # Moderate increase for hourly
            else:
                variability_boost = 1.0  # No change for daily
            
            # Boost the conditional covariance
            cov_cond_W *= variability_boost
            cov_cond_W += np.eye(horizon) * 1e-6
            
            try:
                L = cholesky(cov_cond_W, lower=True)
                z = np.random.normal(size=(horizon, n_sims))
                simulated_paths_W = mean_cond_W[:, np.newaxis] + L @ z
                simulated_paths_log_sigma = nu * simulated_paths_W.T
                mean_forecast = np.mean(simulated_paths_log_sigma, axis=0)
            except np.linalg.LinAlgError:
                print("Warning: Cholesky decomposition failed, using enhanced mean forecast")
                # Add some noise to preserve variability even when using mean forecast
                noise_scale = np.std(past_log_rv_series_truncated) * 0.3
                noise = np.random.normal(0, noise_scale, horizon)
                mean_forecast = nu * mean_cond_W + noise
        else:
            # Even for single simulation, add some variability for high-frequency data
            base_forecast = nu * mean_cond_W
            if freq in ['5T', '5min', 'h', 'H']:
                noise_scale = np.std(past_log_rv_series_truncated) * 0.2
                noise = np.random.normal(0, noise_scale, horizon)
                mean_forecast = base_forecast + noise
            else:
                mean_forecast = base_forecast
    
    except Exception as e:
        print(f"Error in RFSV forecast: {e}")
        # Fallback: use simple persistence with some noise
        last_value = past_log_rv_series_truncated.iloc[-1]
        noise_std = past_log_rv_series_truncated.diff().std()
        mean_forecast = np.full(horizon, last_value) + np.random.normal(0, noise_std, horizon)
    
    # Apply continuity adjustment with gradual decay for high-frequency data
    if use_last_value and len(past_log_rv_series) > 0:
        last_observed = past_log_rv_series.iloc[-1]
        if len(mean_forecast) > 0:
            shift = last_observed - mean_forecast[0]
            
            # For high-frequency data, apply gradual decay of the adjustment
            if freq in ['5T', '5min'] and horizon > 10:
                decay = np.exp(-np.arange(horizon) / (horizon / 5))  # Faster decay for 5-min
            elif freq in ['h', 'H'] and horizon > 5:
                decay = np.exp(-np.arange(horizon) / (horizon / 3))  # Moderate decay for hourly
            else:
                decay = np.ones(horizon)  # No decay for daily or short horizons
            
            mean_forecast = mean_forecast + shift * decay
    
    print(f"RFSV Debug - Output range: {mean_forecast.min():.3f} to {mean_forecast.max():.3f}")

    forecast_series = pd.Series(mean_forecast, name='rfsv_forecast')
    if index is not None:
        forecast_series.index = index
        
    return forecast_series  

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