import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    """
    Creates enhanced bar charts to compare RMSE and QLIKE metrics across models.
    This provides a clear, quantitative comparison of model performance and handles extreme outliers.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    rmse_df.plot(kind='bar', ax=ax1, width=0.8, colormap='viridis')
    ax1.set_title(f'Root Mean Squared Error (RMSE) per Horizon\n(Frequency: {freq})', fontsize=16, weight='bold')
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Models', loc='upper right')
    ax1.set_xlabel('')

    qlike_df.plot(kind='bar', ax=ax2, width=0.8, colormap='plasma')
    ax2.set_title(f'QLIKE Loss per Horizon\n(Frequency: {freq})', fontsize=16, weight='bold')
    ax2.set_xlabel('Forecasting Horizon', fontsize=12)
    ax2.set_ylabel('QLIKE Loss', fontsize=12)
    ax2.tick_params(axis='x', rotation=45, labelsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend(title='Models', loc='upper right')
    
    max_val = qlike_df.max().max()
    p99 = qlike_df.stack().quantile(0.99)
    
    if max_val > p99 * 1.5:
        ax2.set_ylim(top=p99 * 1.2)
        ax2.text(0.98, 0.95, 'Note: Y-axis capped to show detail.\nExtreme outlier(s) not fully shown.',
                 transform=ax2.transAxes,
                 horizontalalignment='right',
                 verticalalignment='top',
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

    plt.tight_layout(h_pad=4)
    
    return fig

def plot_error_diagnostics(actuals, forecasts_dict, freq=''):
    """
    Creates insightful diagnostic plots for forecast errors for each model.
    This replaces the simple boxplot with a more comprehensive analysis to understand
    the nature of the forecast errors.
    
    1. Forecast vs. Actual: Reveals systematic biases (e.g., over/under-prediction).
    2. Error Distribution: Shows if errors are centered around zero and their spread.
    3. Errors vs. Volatility Level: Crucially shows if a model's performance
       degrades in high or low volatility regimes.
    """
    models = list(forecasts_dict.keys())
    n_models = len(models)
    
    # Create a figure with a 3-panel subplot for each model
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 5 * n_models), squeeze=False)
    fig.suptitle(f'Forecast Error Diagnostics (Frequency: {freq})', fontsize=20, weight='bold', y=1.03)

    for i, model_name in enumerate(models):
        forecast_series = forecasts_dict[model_name]
        errors = forecast_series - actuals
        
        # --- Plot 1: Forecast vs. Actual ---
        ax1 = axes[i, 0]
        min_val = min(actuals.min(), forecast_series.min()) * 0.95
        max_val = max(actuals.max(), forecast_series.max()) * 1.05
        ax1.scatter(actuals, forecast_series, alpha=0.4, s=20, edgecolors='k', linewidth=0.5)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Forecast')
        ax1.set_xlabel('Actual Log-RV')
        ax1.set_ylabel('Forecasted Log-RV')
        ax1.set_title(f'{model_name}: Forecast vs. Actual', weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', 'box')

        # --- Plot 2: Error Distribution (KDE) ---
        ax2 = axes[i, 1]
        sns.kdeplot(errors, ax=ax2, fill=True, lw=2)
        ax2.axvline(0, color='r', linestyle='--')
        ax2.set_xlabel('Forecast Error (Forecast - Actual)')
        ax2.set_title(f'{model_name}: Error Distribution', weight='bold')
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Errors vs. Volatility Level ---
        ax3 = axes[i, 2]
        ax3.scatter(actuals, errors, alpha=0.4, s=20, edgecolors='k', linewidth=0.5)
        ax3.axhline(0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Actual Log-RV (Volatility Level)')
        ax3.set_ylabel('Forecast Error')
        ax3.set_title(f'{model_name}: Errors vs. Volatility Level', weight='bold')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def vol_scaler(log_rv_series, freq):
    """
    Transforms log realized volatility to annualized volatility in percentage terms.
    Args:
        log_rv_series (pd.Series): Log realized variance series.
        freq (str): Frequency of the data.
    Returns:
        pd.Series: Annualized volatility in percentage terms.
    """
    ann_factors = {
        'D': 252,
        'h': 252 * 24,
        '5m': 252 * 24 * 12 
    }
    if freq not in ann_factors:
        raise ValueError("Frequency must be one of 'd', 'h', or '5m'")

    factor = ann_factors.get(freq)

    rv = np.exp(log_rv_series)
    rvol = np.sqrt(rv)
    ann_vol_pct = rvol * np.sqrt(factor) * 100
    
    return ann_vol_pct
