import numpy as np
import pandas as pd
import statsmodels.api as sm



def forecast_har_rolling(har_data_complete, horizon, window_size=252, last_log_rv=None):
    """
    Rolling window HAR estimation with 1-step-ahead forecasts.
    
    Args:
        har_data_complete (pd.DataFrame): Complete HAR data (train + test structure)
        horizon (int): Number of 1-step forecasts to make
        window_size (int): Rolling window size (252 for daily, etc.)
        last_log_rv (float): Last observed log realized volatility (for continuity)
    
    Returns:
        pd.Series: 1-step-ahead forecasted log realized volatilities
    """
    forecasts = []
    forecast_dates = []
    
    # Start forecasting from the end of the training period
    start_idx = len(har_data_complete) - horizon
    
    for i in range(horizon):
        current_idx = start_idx + i
        
        if current_idx >= window_size:
            window_data = har_data_complete.iloc[current_idx - window_size:current_idx]
        else:
            window_data = har_data_complete.iloc[:current_idx]
        
        if len(window_data) < 10:
            forecasts.append(np.nan)
            forecast_dates.append(har_data_complete.index[current_idx])
            continue
            
        try:
            # Estimate HAR on the rolling window
            X = window_data[['daily_lag', 'weekly_lag', 'monthly_lag']]
            X = sm.add_constant(X)
            y = window_data['log_rv']
            
            model = sm.OLS(y, X)
            model_fit = model.fit()
            
            # Get the current lags for prediction (from the last observation in window)
            current_lags = window_data[['daily_lag', 'weekly_lag', 'monthly_lag']].iloc[-1]
            X_forecast = pd.DataFrame([current_lags])
            X_forecast = sm.add_constant(X_forecast, has_constant='add')
            
            # Make 1-step-ahead forecast
            next_pred = model_fit.predict(X_forecast).iloc[0]
            
        except Exception as e:
            print(f"Error estimating HAR model: {e}")
        
        forecasts.append(next_pred)
        forecast_dates.append(har_data_complete.index[current_idx])
    
    if last_log_rv is not None and len(forecasts) > 0 and not np.isnan(forecasts[0]):
        shift = last_log_rv - forecasts[0]
        forecasts = [f + shift if not np.isnan(f) else f for f in forecasts]
    
    return pd.Series(forecasts, index=forecast_dates, name='har_forecast')

# ==============================================================================
# Previous HAR estimation function for historical purposes
# ==============================================================================

def estimate_har(har_data):
    """
    Estimate HAR-RV through OLS.

    Args:
        har_data (pandas.DataFrame): Pre-processed data.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Model fit.
    """
    X = har_data[['daily_lag', 'weekly_lag', 'monthly_lag']]
    X = sm.add_constant(X)
    y = har_data['log_rv']
    
    model = sm.OLS(y, X)
    model_fit = model.fit()
    return model_fit

def forecast_har_iterative(model_fit, latest_lags, horizon, last_known_date, freq, last_log_rv=None):
    """
    Legacy iterative HAR forecast - kept for compatibility.
    Use forecast_har_rolling for more realistic rolling window approach.
    """
    current_lags = latest_lags.copy()
    predictions = []

    if 'H' in freq or 'h' in freq:
        weekly_window = 5 * 24
        monthly_window = 22 * 24
    elif '5min' in freq or '5T' in freq:
        weekly_window = 5 * 288
        monthly_window = 22 * 288
    else:
        weekly_window = 5
        monthly_window = 22

    for i in range(horizon):
        X_forecast = pd.DataFrame([current_lags])
        X_forecast = sm.add_constant(X_forecast, has_constant='add')
        next_pred = model_fit.predict(X_forecast).iloc[0]
        
        # Per il primo valore, usa l'ultimo osservato se disponibile
        if i == 0 and last_log_rv is not None:
            predictions.append(last_log_rv)
            # Aggiorna i lag usando il valore reale invece della predizione
            current_lags = pd.Series({
                'daily_lag': last_log_rv,
                'weekly_lag': (current_lags['weekly_lag'] * (weekly_window - 1) + last_log_rv) / weekly_window,
                'monthly_lag': (current_lags['monthly_lag'] * (monthly_window - 1) + last_log_rv) / monthly_window
            })
        else:
            predictions.append(next_pred)
            current_lags = pd.Series({
                'daily_lag': next_pred,
                'weekly_lag': (current_lags['weekly_lag'] * (weekly_window - 1) + next_pred) / weekly_window,
                'monthly_lag': (current_lags['monthly_lag'] * (monthly_window - 1) + next_pred) / monthly_window
            })
    
    return pd.Series(predictions, name='har_forecast')