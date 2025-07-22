import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

def qlike_loss(y_true, y_pred):
    """Calculates the QLIKE loss function."""
    rv_true = np.exp(y_true)
    rv_pred = np.exp(y_pred)
    # Add a small epsilon to avoid log(0) or division by zero
    epsilon = 1e-10
    return np.mean(rv_pred / (rv_true + epsilon) - np.log(rv_pred / (rv_true + epsilon)) - 1)

def evaluate_forecasts(actuals, forecasts_dict):
    """
    Calculates performance metrics (RMSE, QLIKE) for multiple models over a forecast horizon.

    Args:
        actuals (pd.Series): The true future values.
        forecasts_dict (dict): A dictionary of forecast Series, e.g., {'GARCH': forecast_garch, ...}.

    Returns:
        tuple: (DataFrame with RMSE results, DataFrame with QLIKE results).
    """
    horizon = len(actuals)
    models = list(forecasts_dict.keys())
    
    rmse_data = {model: [] for model in models}
    qlike_data = {model: [] for model in models}

    for h in range(horizon):
        y_true_step = actuals.iloc[h]
        for model in models:
            y_pred_step = forecasts_dict[model].iloc[h]
            
            rmse = np.sqrt(mean_squared_error([y_true_step], [y_pred_step]))
            qlike = qlike_loss(np.array([y_true_step]), np.array([y_pred_step]))
            
            rmse_data[model].append(rmse)
            qlike_data[model].append(qlike)
            
    index_labels = [f't+{i+1}' for i in range(horizon)]
    rmse_df = pd.DataFrame(rmse_data, index=index_labels)
    qlike_df = pd.DataFrame(qlike_data, index=index_labels)
    
    return rmse_df, qlike_df

def plot_forecast_comparison(actuals, forecasts_dict, freq):
    """Plots actual values vs. forecasted paths from different models."""
    plt.figure(figsize=(14, 7))
    
    # Plot actual values
    plt.plot(actuals.index, actuals.values, label='Real Value', color='black', linewidth=2.5, marker='o')

    # Plot forecasts from each model
    for model_name, forecast_series in forecasts_dict.items():
        plt.plot(forecast_series.index, forecast_series.values, label=model_name, linestyle='--')

    plt.title(f'Forecasted v Observed (Frequency: {freq})')
    plt.xlabel('Date')
    plt.ylabel('Log Realized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def plot_metrics_comparison(metrics_df, metric_name='RMSE', freq=''):
    """Creates a bar chart to compare a metric across models."""
    metrics_df.plot(kind='bar', figsize=(14, 7), width=0.8)
    plt.title(f'{metric_name} per Horizon (Frequency: {freq})')
    plt.xlabel('Forecasting Horizon')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()