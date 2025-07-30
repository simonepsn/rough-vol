import numpy as np
import pandas as pd
import statsmodels.api as sm


def forecast_har_rolling(har_data_complete, horizon, window_size=252, last_log_rv=None):
    """
    Rolling window HAR estimation with 1-step-ahead forecasts.
    Simplified version that uses simple lag logic.
    """
    forecasts = []
    forecast_dates = []
    
    # Start forecasting from the end of the training period
    start_idx = len(har_data_complete) - horizon
    
    for i in range(horizon):
        current_idx = start_idx + i
        
        # Get training window
        if current_idx >= window_size:
            window_data = har_data_complete.iloc[current_idx - window_size:current_idx]
        else:
            window_data = har_data_complete.iloc[:current_idx]
            
        try:
            # Estimate HAR on the rolling window
            X = window_data[['daily_lag', 'weekly_lag', 'monthly_lag']].copy()
            y = window_data['log_rv'].copy()
            
            X_with_const = sm.add_constant(X, has_constant='add')
            model = sm.OLS(y, X_with_const)
            model_fit = model.fit()
            
            # Get forecast date
            forecast_date = har_data_complete.index[current_idx]
            
            # Calculate lags using simplified logic
            current_lags = calculate_calendar_lags(
                har_data_complete.iloc[:current_idx], 
                forecast_date
            )
            
            X_forecast = pd.DataFrame([current_lags], columns=['daily_lag', 'weekly_lag', 'monthly_lag'])
            X_forecast_with_const = sm.add_constant(X_forecast, has_constant='add')
            
            # Make forecast
            next_pred = model_fit.predict(X_forecast_with_const).iloc[0]
            
        except Exception as e:
            print(f"Error estimating HAR model at step {i}: {e}")
            next_pred = window_data['log_rv'].mean()
        
        forecasts.append(next_pred)
        forecast_dates.append(forecast_date)
    
    # Apply continuity adjustment
    if last_log_rv is not None and len(forecasts) > 0:
        shift = last_log_rv - forecasts[0]
        forecasts = [f + shift for f in forecasts]
    
    return pd.Series(forecasts, index=forecast_dates, name='har_forecast')


def calculate_calendar_lags(har_data, forecast_date):
    """
    Calculate HAR lags based on calendar dates from the time series index.
    
    Args:
        har_data (pd.DataFrame): Historical HAR data with DateTimeIndex
        forecast_date (pd.Timestamp): Date for which we're forecasting
        
    Returns:
        pd.Series: Dictionary with daily_lag, weekly_lag, monthly_lag
    """
    # Get the log_rv series with datetime index
    log_rv_series = har_data['log_rv']
    
    # Initialize default values
    daily_lag = 0.0
    weekly_lag = 0.0
    monthly_lag = 0.0
    
    # Determine frequency from the index
    freq = pd.infer_freq(har_data.index)
    
    if freq is None:
        # Fallback: estimate frequency from median time difference
        time_diffs = har_data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            
            if median_diff <= pd.Timedelta('6 minutes'):
                freq_type = '5min'
            elif median_diff <= pd.Timedelta('2 hours'):
                freq_type = 'hourly'
            else:
                freq_type = 'daily'
        else:
            freq_type = 'daily'  # Default fallback
    else:
        if 'T' in freq or 'min' in freq.lower():
            freq_type = '5min'
        elif 'H' in freq or 'h' in freq.lower():
            freq_type = 'hourly'
        else:
            freq_type = 'daily'
    
    # Calculate lags based on frequency
    try:
        if freq_type == 'daily':  # Corretto da 'D' a 'daily'
            # Daily: 1 day, 5 days (weekly), 22 days (monthly)
            daily_lag = get_lag_value(log_rv_series, forecast_date, days=1)
            weekly_lag = get_rolling_average(log_rv_series, forecast_date, days=5)
            monthly_lag = get_rolling_average(log_rv_series, forecast_date, days=22)
            
        elif freq_type == 'hourly':  # Corretto da 'h' a 'hourly'
            # Hourly: 1 hour, 5*24 hours (weekly), 22*24 hours (monthly)
            daily_lag = get_lag_value(log_rv_series, forecast_date, hours=1)
            weekly_lag = get_rolling_average(log_rv_series, forecast_date, hours=5*24)
            monthly_lag = get_rolling_average(log_rv_series, forecast_date, hours=22*24)
            
        elif freq_type == '5min':
            # 5-minute: 1 period, 5*288 periods (weekly), 22*288 periods (monthly)
            daily_lag = get_lag_value(log_rv_series, forecast_date, minutes=5)
            weekly_lag = get_rolling_average(log_rv_series, forecast_date, minutes=5*288)
            monthly_lag = get_rolling_average(log_rv_series, forecast_date, minutes=5*288*22)
    
    except Exception as e:
        print(f"Error calculating lags: {e}")
        # Use fallback values
        if len(log_rv_series) > 0:
            daily_lag = log_rv_series.iloc[-1]
            weekly_lag = log_rv_series.iloc[-min(5, len(log_rv_series)):].mean()
            monthly_lag = log_rv_series.iloc[-min(22, len(log_rv_series)):].mean()
    
    return pd.Series({
        'daily_lag': daily_lag,
        'weekly_lag': weekly_lag,
        'monthly_lag': monthly_lag
    })


def get_lag_value(series, forecast_date, days=0, hours=0, minutes=0):
    """
    Get lagged value based on calendar time.
    """
    lag_date = forecast_date - pd.Timedelta(days=days, hours=hours, minutes=minutes)
    
    # Find the closest available date
    available_dates = series.index[series.index <= lag_date]
    
    if len(available_dates) == 0:
        return series.iloc[0] if len(series) > 0 else 0.0
    
    closest_date = available_dates[-1]
    return series.loc[closest_date]


def get_rolling_average(series, forecast_date, days=0, hours=0, minutes=0):
    """
    Get rolling average over the specified period ending at lag_date.
    """
    lag_date = forecast_date - pd.Timedelta(days=days, hours=hours, minutes=minutes)
    end_date = forecast_date - pd.Timedelta(days=0, hours=0, minutes=5 if minutes > 0 else (1 if hours > 0 else 0))
    
    # Get data in the rolling window
    window_data = series[(series.index >= lag_date) & (series.index <= end_date)]
    
    if len(window_data) == 0:
        # Fallback to available data
        available_data = series[series.index <= end_date]
        if len(available_data) == 0:
            return series.iloc[0] if len(series) > 0 else 0.0
        else:
            return available_data.iloc[-min(len(available_data), 5):].mean()
    
    return window_data.mean()


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