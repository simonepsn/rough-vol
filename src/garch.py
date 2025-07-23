import numpy as np
import pandas as pd
from arch import arch_model

def forecast_garch_rolling(log_returns_series, horizon, window_size=252, last_log_rv=None):
    """
    Rolling window GARCH estimation with 1-step-ahead forecasts.
    
    Args:
        log_returns_series (pd.Series): Complete log returns series (train + test)
        horizon (int): Number of 1-step forecasts to make
        window_size (int): Rolling window size (252 for daily, 252*24 for hourly, etc.)
        last_log_rv (float): Last observed log realized volatility (for continuity)
    
    Returns:
        pd.Series: 1-step-ahead forecasted log realized volatilities
    """
    forecasts = []
    forecast_dates = []
    
    # Start forecasting from the end of the training period
    start_idx = len(log_returns_series) - horizon
    
    for i in range(horizon):

        current_idx = start_idx + i
        
        if current_idx >= window_size:
            window_returns = log_returns_series.iloc[current_idx - window_size:current_idx]
        else:
            window_returns = log_returns_series.iloc[:current_idx]
        
        # Skip if window is too small
        if len(window_returns) < 10:
            forecasts.append(np.nan)
            forecast_dates.append(log_returns_series.index[current_idx])
            continue
        
        # Estimate GARCH on the rolling window
        try:
            model = arch_model(window_returns * 100, vol='Garch', p=1, q=1, dist='Normal')
            model_fit = model.fit(update_freq=0, disp='off')
            
            forecast_result = model_fit.forecast(horizon=1, reindex=False)
            predicted_variance = forecast_result.variance.iloc[0, 0]
            
            log_predicted_variance = np.log(predicted_variance / (100**2))
            
        except Exception as e:
            print(f"Error estimating GARCH model: {e}")
        
        forecasts.append(log_predicted_variance)
        forecast_dates.append(log_returns_series.index[current_idx])

    forecasts = pd.Series(forecasts, index=forecast_dates, name='garch_forecast')
    forecasts.index = pd.to_datetime(forecasts.index, errors='coerce') 
    
    # Apply continuity adjustment to first forecast if last_log_rv is provided
    if last_log_rv is not None and len(forecasts) > 0:
        shift = last_log_rv - forecasts[0]
        forecasts = [f + shift for f in forecasts]
    
    return pd.Series(forecasts, index=forecast_dates, name='garch_forecast')

# ===========================================================================
# Previous versions, which are not recommended for rolling forecasts:
# ===========================================================================
def estimate_garch(log_returns_series):
    """
    Estimate a GARCH(1,1) on log returns,
    log-returns * 100 is common practice for numerical stability.

    Args:
        log_returns_series (pandas.Series): Daily log-returns.

    Returns:
        arch.univariate.base.ARCHModelResult: Final model object.
    """

    model = arch_model(log_returns_series * 100, vol='Garch', p=1, q=1, dist='Normal')

    # Add a step in which we add a block for the case in which the model does not converge
    try:
        model_fit = model.fit(update_freq=0, disp='off')
    except Exception as e:
        return np.nan
     
    return model_fit

def forecast_garch(model_fit, horizon, last_known_date, freq, last_log_rv=None):
    """
    Legacy function - kept for compatibility but not recommended for rolling forecasts.
    Use forecast_garch_rolling instead.
    """
    
    forecast = model_fit.forecast(horizon=horizon, reindex=False)
    predicted_variances_values = forecast.variance.iloc[0].values
    log_predicted_variances = np.log(predicted_variances_values / (100**2))
    
    forecast_index = pd.date_range(
        start=last_known_date + pd.tseries.frequencies.to_offset(freq), 
        periods=horizon, 
        freq=freq
    )    
    
    if last_log_rv is not None:
        shift = last_log_rv - log_predicted_variances[0]
        log_predicted_variances = log_predicted_variances + shift

    return pd.Series(log_predicted_variances, index=forecast_index, name='garch_forecast')