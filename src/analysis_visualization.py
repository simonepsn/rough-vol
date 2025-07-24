import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error

def filter_trading_hours(data_series, freq='5min'):
    """
    Filter data to include only trading hours (9:30 AM - 4:00 PM EST) for intraday frequencies.
    Also removes weekends.
    """
    if freq not in ['5min', '5T', '5m', 'h', 'H', 'Hourly']:
        return data_series  # No filtering for daily data
    
    # Create a copy to avoid modifying original
    filtered_data = data_series.copy()
    
    # Remove weekends
    filtered_data = filtered_data[filtered_data.index.weekday < 5]
    
    # For intraday data, filter trading hours (9:30 AM - 4:00 PM EST)
    if freq in ['5min', '5T', '5m']:
        # Keep only 9:30 AM to 4:00 PM
        filtered_data = filtered_data.between_time('09:30', '16:00')
    elif freq in ['h', 'H', 'Hourly']:
        # For hourly, keep 9:00 AM to 4:00 PM to capture full trading day
        filtered_data = filtered_data.between_time('09:00', '16:00')
    
    # Remove periods with extremely low activity (likely half-days or holidays)
    if len(filtered_data) > 100:  # Only if we have enough data
        # Group by date and count observations per day
        daily_counts = filtered_data.groupby(filtered_data.index.date).count()
        
        if freq in ['5min', '5T', '5m']:
            min_obs_per_day = 50  # Expect ~78 observations per trading day (6.5 hours * 12)
        elif freq in ['h', 'H', 'Hourly']:
            min_obs_per_day = 5   # Expect ~8 observations per trading day
        
        # Keep only dates with sufficient observations
        valid_dates = daily_counts[daily_counts >= min_obs_per_day].index
        date_filter = pd.Series(filtered_data.index.date).isin(valid_dates)
        filtered_data = filtered_data[date_filter.values]
    
    print(f"Trading hours filter: {len(data_series)} -> {len(filtered_data)} observations")
    return filtered_data

def qlike_loss(y_true, y_pred):
    """Calculates the QLIKE loss function safely."""
    # Clip extreme values to prevent overflow
    y_true_clipped = np.clip(y_true, -20, 5)  # Reasonable range for log-RV
    y_pred_clipped = np.clip(y_pred, -20, 5)
    
    rv_true = np.exp(y_true_clipped)
    rv_pred = np.exp(y_pred_clipped)
    
    epsilon = 1e-8
    ratio = rv_pred / (rv_true + epsilon)
    
    # Clip ratio to prevent log(0) or extreme values
    ratio = np.clip(ratio, epsilon, 1e8)
    
    qlike = ratio - np.log(ratio) - 1
    
    # Additional safety check
    qlike = np.clip(qlike, 0, 100)  # QLIKE should be non-negative and reasonable
    
    return np.mean(qlike)

def evaluate_forecasts(actuals, forecasts_dict, freq='D'):
    """
    Calculates performance metrics (RMSE, QLIKE) for multiple models over a forecast horizon.
    Uses index alignment and trading hours filtering to automatically exclude non-trading periods.

    Args:
        actuals (pd.Series): The true future values.
        forecasts_dict (dict): A dictionary of forecast Series.
        freq (str): Data frequency for proper filtering.

    Returns:
        tuple: (DataFrame with summary results, DataFrame with RMSE results, DataFrame with QLIKE results).
    """
    # Apply trading hours filter to actuals
    actuals_filtered = filter_trading_hours(actuals, freq)
    
    # Get common index (intersection of all series) after filtering
    common_index = actuals_filtered.index
    for model, forecast in forecasts_dict.items():
        forecast_filtered = filter_trading_hours(forecast, freq)
        common_index = common_index.intersection(forecast_filtered.index)
    
    # Align all series to common index
    actuals_aligned = actuals_filtered.loc[common_index]
    forecasts_aligned = {}
    for model, forecast in forecasts_dict.items():
        forecast_filtered = filter_trading_hours(forecast, freq)
        forecasts_aligned[model] = forecast_filtered.loc[common_index]
    
    print(f"Using {len(common_index)} common trading periods for evaluation")
    
    horizon = len(actuals_aligned)
    models = list(forecasts_aligned.keys())
    
    rmse_data = {model: [] for model in models}
    qlike_data = {model: [] for model in models}

    for h in range(horizon):
        y_true_step = actuals_aligned.iloc[h]
        for model in models:
            y_pred_step = forecasts_aligned[model].iloc[h]
            
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
    """Plots actual values vs. forecasted paths from different models with proper filtering."""
    # Apply trading hours filter
    actuals_filtered = filter_trading_hours(actuals, freq)
    forecasts_filtered = {}
    
    # Get common index after filtering
    common_index = actuals_filtered.index
    for model_name, forecast_series in forecasts_dict.items():
        forecast_filtered = filter_trading_hours(forecast_series, freq)
        forecasts_filtered[model_name] = forecast_filtered
        common_index = common_index.intersection(forecast_filtered.index)
    
    # Align to common index
    actuals_plot = actuals_filtered.loc[common_index]
    forecasts_plot = {model: forecasts_filtered[model].loc[common_index] 
                     for model in forecasts_filtered.keys()}
    
    plt.figure(figsize=(14, 7))
    
    # For high-frequency data, use different plotting approach
    if freq in ['5min', '5T', '5m'] and len(actuals_plot) > 1000:
        # Sample data for visualization (every 20th point for 5-min data)
        sample_freq = max(1, len(actuals_plot) // 500)  # Show ~500 points max
        actuals_sample = actuals_plot.iloc[::sample_freq]
        forecasts_sample = {model: series.iloc[::sample_freq] 
                          for model, series in forecasts_plot.items()}
        
        plt.plot(actuals_sample.index, actuals_sample.values, 
                label='Real Value', color='black', linewidth=1.5, alpha=0.8)
        
        for model_name, forecast_series in forecasts_sample.items():
            plt.plot(forecast_series.index, forecast_series.values, 
                    label=model_name, linestyle='--', alpha=0.7)
        
        plt.title(f'Forecasted vs Observed (Frequency: {freq}) - Sampled Data')
        
    elif freq in ['h', 'H', 'Hourly'] and len(actuals_plot) > 200:
        # Sample hourly data if too dense
        sample_freq = max(1, len(actuals_plot) // 200)
        actuals_sample = actuals_plot.iloc[::sample_freq]
        forecasts_sample = {model: series.iloc[::sample_freq] 
                          for model, series in forecasts_plot.items()}
        
        plt.plot(actuals_sample.index, actuals_sample.values, 
                label='Real Value', color='black', linewidth=2, marker='o', markersize=3)
        
        for model_name, forecast_series in forecasts_sample.items():
            plt.plot(forecast_series.index, forecast_series.values, 
                    label=model_name, linestyle='--', marker='s', markersize=2)
        
        plt.title(f'Forecasted vs Observed (Frequency: {freq}) - Sampled Data')
        
    else:
        # Daily data or small datasets - plot all points
        plt.plot(actuals_plot.index, actuals_plot.values, 
                label='Real Value', color='black', linewidth=2.5, marker='o')

        for model_name, forecast_series in forecasts_plot.items():
            plt.plot(forecast_series.index, forecast_series.values, 
                    label=model_name, linestyle='--')
        
        plt.title(f'Forecasted vs Observed (Frequency: {freq})')

    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
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
    Applies trading hours filtering for intraday data.
    """
    # Apply trading hours filter
    actuals_filtered = filter_trading_hours(actuals, freq)
    forecasts_filtered = {}
    
    # Get common index after filtering
    common_index = actuals_filtered.index
    for model_name, forecast_series in forecasts_dict.items():
        forecast_filtered = filter_trading_hours(forecast_series, freq)
        forecasts_filtered[model_name] = forecast_filtered
        common_index = common_index.intersection(forecast_filtered.index)
    
    # Align to common index
    actuals_plot = actuals_filtered.loc[common_index]
    forecasts_plot = {model: forecasts_filtered[model].loc[common_index] 
                     for model in forecasts_filtered.keys()}
    
    models = list(forecasts_plot.keys())
    n_models = len(models)
    
    # Create a figure with a 3-panel subplot for each model
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 5 * n_models), squeeze=False)
    fig.suptitle(f'Forecast Error Diagnostics (Frequency: {freq})', fontsize=20, weight='bold', y=1.03)

    for i, model_name in enumerate(models):
        forecast_series = forecasts_plot[model_name]
        errors = forecast_series - actuals_plot
        
        # --- Plot 1: Forecast vs. Actual ---
        ax1 = axes[i, 0]
        
        # For high-frequency data, sample for better visualization
        if len(actuals_plot) > 1000:
            sample_freq = max(1, len(actuals_plot) // 1000)
            actuals_sample = actuals_plot.iloc[::sample_freq]
            forecast_sample = forecast_series.iloc[::sample_freq]
        else:
            actuals_sample = actuals_plot
            forecast_sample = forecast_series
        
        min_val = min(actuals_sample.min(), forecast_sample.min()) * 0.95
        max_val = max(actuals_sample.max(), forecast_sample.max()) * 1.05
        ax1.scatter(actuals_sample, forecast_sample, alpha=0.4, s=20, edgecolors='k', linewidth=0.5)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Forecast')
        ax1.set_xlabel('Actual Volatility (%)')
        ax1.set_ylabel('Forecasted Volatility (%)')
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
        
        # Sample for visualization if needed
        if len(errors) > 1000:
            sample_freq = max(1, len(errors) // 1000)
            errors_sample = errors.iloc[::sample_freq]
            actuals_sample = actuals_plot.iloc[::sample_freq]
        else:
            errors_sample = errors
            actuals_sample = actuals_plot
            
        ax3.scatter(actuals_sample, errors_sample, alpha=0.4, s=20, edgecolors='k', linewidth=0.5)
        ax3.axhline(0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Actual Volatility Level (%)')
        ax3.set_ylabel('Forecast Error')
        ax3.set_title(f'{model_name}: Errors vs. Volatility Level', weight='bold')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def vol_scaler(log_rv_series, freq):
    """
    Transforms log realized volatility to annualized volatility in percentage terms.
    Uses more robust scaling for high-frequency data to prevent unrealistic values.
    Args:
        log_rv_series (pd.Series): Log realized variance series.
        freq (str): Frequency of the data.
    Returns:
        pd.Series: Annualized volatility in percentage terms.
    """
    ann_factors = {
        'D': 252,
        'Daily': 252,
        'h': 252 * 24,
        'H': 252 * 24,
        'Hourly': 252 * 24,
        '5m': 252 * 24 * 12,
        '5min': 252 * 24 * 12,
        '5T': 252 * 24 * 12,
        '5-Minutes': 252 * 24 * 12
    }
    
    if freq not in ann_factors:
        print(f"Warning: Frequency '{freq}' not recognized. Available: {list(ann_factors.keys())}")
        factor = 252  # Default to daily
    else:
        factor = ann_factors[freq]

    # More aggressive clipping for high-frequency data
    if freq in ['5m', '5min', '5T', '5-Minutes']:
        # For 5-minute data, use tighter bounds based on your actual data range
        log_rv_clipped = np.clip(log_rv_series, -18, -10)  # More realistic range for 5-min
        print(f"5-min data: Original range [{log_rv_series.min():.3f}, {log_rv_series.max():.3f}] -> Clipped range [{log_rv_clipped.min():.3f}, {log_rv_clipped.max():.3f}]")
    elif freq in ['h', 'H', 'Hourly']:
        # For hourly data
        log_rv_clipped = np.clip(log_rv_series, -16, -8)  # More realistic range for hourly
        print(f"Hourly data: Original range [{log_rv_series.min():.3f}, {log_rv_series.max():.3f}] -> Clipped range [{log_rv_clipped.min():.3f}, {log_rv_clipped.max():.3f}]")
    else:
        # For daily data, keep original bounds
        log_rv_clipped = np.clip(log_rv_series, -20, 5)
    
    rv = np.exp(log_rv_clipped)
    rvol = np.sqrt(rv)
    ann_vol_pct = rvol * np.sqrt(factor) * 100
    
    # Additional sanity check: cap at reasonable volatility levels
    if freq in ['5m', '5min', '5T', '5-Minutes']:
        ann_vol_pct = np.clip(ann_vol_pct, 0, 200)  # Cap at 200% annualized vol for 5-min
    elif freq in ['h', 'H', 'Hourly']:
        ann_vol_pct = np.clip(ann_vol_pct, 0, 150)  # Cap at 150% annualized vol for hourly
    else:
        ann_vol_pct = np.clip(ann_vol_pct, 0, 100)   # Cap at 100% annualized vol for daily
    
    return ann_vol_pct
