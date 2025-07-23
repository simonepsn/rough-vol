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
    
    rmse_mean = rmse_df.mean()
    qlike_mean = qlike_df.mean()
    
    results_summary = pd.DataFrame({
        'RMSE': rmse_mean,
        'QLIKE': qlike_mean
    })
    
    return results_summary, rmse_df, qlike_df

def plot_forecast_comparison(actuals, forecasts_dict, freq):
    """Plots actual values vs. forecasted paths from different models."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(actuals.index, actuals.values, label='Real Value', color='black', linewidth=2.5, marker='o')

    for model_name, forecast_series in forecasts_dict.items():
        plt.plot(forecast_series.index, forecast_series.values, label=model_name, linestyle='--')

    plt.title(f'Forecasted v Observed (Frequency: {freq})')
    plt.xlabel('Date')
    plt.ylabel('Log Realized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    return plt.gcf()

def plot_metrics_comparison(rmse_df, qlike_df, freq=''):
    """Creates bar charts to compare RMSE and QLIKE metrics across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    rmse_df.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'RMSE per Horizon (Frequency: {freq})')
    ax1.set_xlabel('Forecasting Horizon')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    qlike_df.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title(f'QLIKE per Horizon (Frequency: {freq})')
    ax2.set_xlabel('Forecasting Horizon')
    ax2.set_ylabel('QLIKE')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_error_analysis(actuals, forecasts_dict, freq=''):
    """Creates boxplots to analyze forecast errors across models."""
    errors_df = pd.DataFrame()
    
    for model_name, forecast_series in forecasts_dict.items():
        errors_df[model_name] = forecast_series.values - actuals.values
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    box_plot = errors_df.boxplot(ax=ax, grid=False, return_type='dict')
    
    ax.set_title(f"Forecast Error Analysis (Frequency: {freq})", fontsize=16, weight='bold')
    ax.set_ylabel('Forecasting Error', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Bias Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def vol_scaler(log_rv_series, freq):
    """
    Transforms log realized volatility to annualized volatility in percentage terms.
    Args:
        log_rv_series (pd.Series): Log realized variance series.
        freq (str): Frequency of the data.
    Returns:
        pd.Series: Annualized volatility in percentage terms.
    """
    rv = np.exp(log_rv_series)
    rvol = np.sqrt(rv)
    ann_vol_pct = rvol * np.sqrt(252) * 100
    
    return ann_vol_pct
