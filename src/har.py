import numpy as np
import pandas as pd
import statsmodels.api as sm

def prepare_har_data(log_rv_series, freq='D'):
    """
    Prepare HAR-df for analysis
    """
    if freq == 'D':
        periods_in_day = 1
    elif freq == 'H':
        periods_in_day = 24
    elif freq == '5min':
        periods_in_day = 24 * 12
    else:
        raise ValueError("Frequency not yet available. Use 'D', 'H', or '5min'.")
    
    weekly_window = periods_in_day * 5
    monthly_window = periods_in_day * 22
    
    df = pd.DataFrame({'log_rv': log_rv_series})
    
    df['daily_lag'] = df['log_rv'].shift(1)
    df['weekly_lag'] = df['log_rv'].rolling(window=weekly_window).mean().shift(1)
    df['monthly_lag'] = df['log_rv'].rolling(window=monthly_window).mean().shift(1)
    
    df.dropna(inplace=True)
    return df

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

def forecast_har_iterative(model_fit, latest_lags, horizon, last_known_date, freq):
    """
    Iterates a 'horizon'-step ahead estimate for HAR model, outputs a series with timestamp.
    """
    current_lags = latest_lags.copy()
    predictions = []

    weekly_window = 5 
    monthly_window = 22

    for _ in range(horizon):
        X_forecast = pd.DataFrame([current_lags])
        X_forecast = sm.add_constant(X_forecast, has_constant='add')
        next_pred = model_fit.predict(X_forecast).iloc[0]
        predictions.append(next_pred)
        
        new_daily_val = next_pred
        new_weekly_lag = (current_lags['daily_lag'] * (weekly_window - 1) + new_daily_val) / weekly_window
        new_monthly_lag = (current_lags['weekly_lag'] * (monthly_window - 1) + new_daily_val) / monthly_window
        
        current_lags = pd.Series({
            'daily_lag': new_daily_val,
            'weekly_lag': new_weekly_lag,
            'monthly_lag': new_monthly_lag
        })
        
    forecast_index = pd.date_range(start=last_known_date, periods=horizon + 1, freq=freq, inclusive='right')
    
    return pd.Series(predictions, index=forecast_index, name='har_forecast')