import numpy as np
import pandas as pd
from arch import arch_model

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
    model_fit = model.fit(disp='off')
    return model_fit

def forecast_garch(model_fit, horizon, last_known_date, freq):
    """
    Forecasts volatility 'horizon'-step ahead, sereis with timestamp.
    """
    forecast = model_fit.forecast(horizon=horizon, reindex=False)
    
    predicted_variances_values = forecast.variance.iloc[0].values
    
    log_predicted_variances = np.log(predicted_variances_values / (100**2))
    
    forecast_index = pd.date_range(start=last_known_date, periods=horizon + 1, freq=freq, inclusive='right')
    
    return pd.Series(log_predicted_variances, index=forecast_index, name='garch_forecast')