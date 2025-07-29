import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error


def evaluate_forecasts(actuals, forecasts_df, freq='D'):
    """
    Calculates performance metrics (RMSE) for multiple models over a forecast horizon.
    Uses index alignment and trading hours filtering to automatically exclude non-trading periods.

    Args:
        actuals (pd.Series): The true future values.
        forecasts_df (pd.DataFrame): A DataFrame where each column represents forecasts
                                      from a different model/ticker combination.
                                      Each column name should represent a unique model/ticker.
        freq (str): Data frequency for proper filtering.

    Returns:
        tuple: (DataFrame with summary RMSE results, DataFrame with RMSE results).
    """
    
    # Get common index (intersection of all series)
    common_index = actuals.index.intersection(forecasts_df.index)

    # Align all series to common index
    actuals_aligned = actuals.loc[common_index]
    forecasts_aligned_df = forecasts_df.loc[common_index] # This line expects forecasts_df to be a DataFrame
        
    horizon = len(actuals_aligned)
    # The models are now correctly identified as the column names of the aligned forecasts DataFrame
    models = forecasts_aligned_df.columns.tolist() 
    
    rmse_data = {model: [] for model in models}

    # Iterate through each step of the forecast horizon
    for h in range(horizon):
        y_true_step = actuals_aligned.iloc[h]
        
        # For each model (column) in the forecasts DataFrame
        for model_col_name in models:
            # Access the specific forecast value for the current step and current model column
            y_pred_step = forecasts_aligned_df[model_col_name].iloc[h] 
            
            rmse = np.sqrt(mean_squared_error([y_true_step], [y_pred_step]))
            
            rmse_data[model_col_name].append(rmse)
            
    index_labels = [f't+{i+1}' for i in range(horizon)]
    rmse_df = pd.DataFrame(rmse_data, index=index_labels)
    
    rmse_mean = rmse_df.mean()
    
    results_summary = pd.DataFrame({
        'RMSE': rmse_mean
    })
    
    return results_summary, rmse_df


def plot_forecast_comparison(actuals: pd.Series, forecasts_dict: dict, freq: str, ticker_name: str = "Unknown Ticker"):
    """
    Plots actual values vs. forecasted paths from different models with robust filtering and error handling.
    Includes checks for empty or all-NaN data to prevent plotting errors.

    Args:
        actuals (pd.Series): The true future values (e.g., actual log realized volatility).
        forecasts_dict (dict): A dictionary where keys are model names (str) and values are
                               pd.Series representing forecasts from that model.
        freq (str): Data frequency (e.g., 'D', 'h', '5min') for proper plotting and scaling.
        ticker_name (str): Name of the ticker for plot titles and warnings.
    
    Returns:
        matplotlib.figure.Figure or None: The generated matplotlib Figure object if plotting is successful,
                                          otherwise None.
    """
    # Ensure actuals is a Series with a name for better warning messages
    if not hasattr(actuals, 'name') or actuals.name is None:
        actuals.name = "Actuals"

    # Filter out models with no valid forecasts
    valid_forecasts_dict = {}
    for model_name, forecast_series in forecasts_dict.items():
        if isinstance(forecast_series, pd.Series) and not forecast_series.empty and not forecast_series.isna().all():
            valid_forecasts_dict[model_name] = forecast_series
        else:
            print(f"Warning: Model '{model_name}' for {ticker_name} ({freq}) has no valid forecast data. Skipping.")

    if not valid_forecasts_dict:
        print(f"Warning: No valid forecast models to plot for {ticker_name} ({freq}). Skipping plot_forecast_comparison.")
        return None

    # Get common index among actuals and all valid forecasts
    common_index = actuals.index
    for forecast_series in valid_forecasts_dict.values():
        common_index = common_index.intersection(forecast_series.index)
    
    if common_index.empty:
        print(f"Warning: No common index found between actuals and forecasts for {ticker_name} ({freq}). Skipping plot_forecast_comparison.")
        return None


    actuals_plot = actuals.loc[common_index]
    forecasts_plot = {model: s.loc[common_index] for model, s in valid_forecasts_dict.items()}

    # Final check: if actuals_plot is empty or all NaN after alignment, skip plotting
    if actuals_plot.empty or actuals_plot.isna().all():
        print(f"Warning: Actuals data is empty or all NaN for {ticker_name} ({freq}) after alignment. Skipping plot_forecast_comparison.")
        return None


    # Sample for better visualization if needed
    sample_freq = 1
    if freq in ['5min', '5T', '5m'] and len(actuals_plot) > 1000:
        sample_freq = max(1, len(actuals_plot) // 500)
    elif freq in ['h', 'H', 'Hourly'] and len(actuals_plot) > 200:
        sample_freq = max(1, len(actuals_plot) // 200)

    actuals_sample = actuals_plot.iloc[::sample_freq]
    forecasts_sample = {m: s.iloc[::sample_freq] for m, s in forecasts_plot.items()}

    
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(actuals_sample.index, actuals_sample.values, label='Real Value', color='black', linewidth=1.5)

    for model_name, series in forecasts_sample.items():
        ax.plot(series.index, series.values, linestyle='--', label=model_name, alpha=0.8)

    ax.set_title(f'{ticker_name} - Forecast vs Actual ({freq})', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility (%)')
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def simple_evaluate_single_ticker(actuals: pd.Series, forecasts_dict: dict) -> pd.DataFrame:
    """
    Performs a robust evaluation (RMSE only) for a single ticker across multiple models.
    Includes checks for data validity (length, NaNs).

    Args:
        actuals (pd.Series): The true "future" values.
        forecasts_dict (dict): A dictionary where keys are model names (str) and values are
                               pd.Series representing forecasts from that model.

    Returns:
        pd.DataFrame: A DataFrame with RMSE results for each model, indexed by model name.
                      Returns an empty DataFrame if no valid data for evaluation.
    """
    results = {}
    
    # Ensure actuals is a Series with a name for better warning messages
    if not hasattr(actuals, 'name') or actuals.name is None:
        actuals.name = "Actuals"

    if actuals.empty or actuals.isna().all():
        print(f"Warning: Actuals data for {actuals.name} is empty or all NaN. Cannot evaluate.")
        return pd.DataFrame(columns=['RMSE'])

    for model_name, forecast_series in forecasts_dict.items():
        try:
            # Ensure forecast_series is a pandas Series
            if not isinstance(forecast_series, pd.Series):
                print(f"  Warning: Forecast for model '{model_name}' is not a pandas Series. Skipping.")
                results[model_name] = {'RMSE': np.nan}
                continue

            # Align actuals and forecasts based on their index
            common_index = actuals.index.intersection(forecast_series.index)
            
            actuals_aligned = actuals.loc[common_index]
            forecast_aligned = forecast_series.loc[common_index]

            # Check for sufficient and non-NaN data after alignment
            if actuals_aligned.empty or forecast_aligned.empty:
                print(f"  Warning: No common data points for {model_name}. Skipping evaluation.")
                results[model_name] = {'RMSE': np.nan}
                continue

            if actuals_aligned.isna().any() or forecast_aligned.isna().any():
                print(f"  Warning: {model_name} has NaN values after alignment. RMSE might be affected or calculation skipped.")
                # Option 1: Drop NaNs (if you want to evaluate on available non-NaN data)
                valid_data_mask = actuals_aligned.notna() & forecast_aligned.notna()
                actuals_clean = actuals_aligned[valid_data_mask]
                forecast_clean = forecast_aligned[valid_data_mask]

                if actuals_clean.empty:
                    print(f"  Warning: No valid non-NaN data points for {model_name} after cleaning. Skipping evaluation.")
                    results[model_name] = {'RMSE': np.nan}
                    continue
            else:
                actuals_clean = actuals_aligned
                forecast_clean = forecast_aligned

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(actuals_clean, forecast_clean))
            results[model_name] = {'RMSE': rmse}
            
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}. Setting RMSE to NaN.")
            results[model_name] = {'RMSE': np.nan}
    
    return pd.DataFrame(results).T

def plot_metrics_comparison(rmse_df, freq=''):
    """
    Creates enhanced bar charts to compare RMSE metrics across models.
    This provides a clear, quantitative comparison of model performance.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7)) # Only one subplot for RMSE
    
    rmse_df.plot(kind='bar', ax=ax1, width=0.8, colormap='viridis')
    ax1.set_title(f'Root Mean Squared Error (RMSE) per Horizon\n(Frequency: {freq})', fontsize=16, weight='bold')
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Models', loc='upper right')
    ax1.set_xlabel('Forecasting Horizon', fontsize=12) # Changed x-label for clarity

    plt.tight_layout()
    
    return fig

def plot_error_diagnostics(actuals, forecasts_dict, freq=''):
    """
    Creates insightful diagnostic plots for forecast errors for each model.
    Applies trading hours filtering for intraday data.
    """
    # Apply trading hours filter
    forecasts = {}
    
    # Get common index after filtering
    common_index = actuals.index
    for model_name, forecast_series in forecasts_dict.items():
        forecasts[model_name] = forecast_series
        common_index = common_index.intersection(forecast_series.index)
    
    # Align to common index
    actuals_plot = actuals.loc[common_index]
    forecasts_plot = {model: forecasts[model].loc[common_index] 
                     for model in forecasts.keys()}
    
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

def vol_scaler(log_rv_df, freq, modeltype):
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
        'D': 250,
        'Daily': 250,
        'h': 250 * 6,
        'H': 250 * 6,
        'Hourly': 250 * 6,
        '5m': 250 * 6 * 12,
        '5min': 250 * 6 * 12,
        '5T': 250 * 6 * 12,
        '5-Minutes': 250 * 6 * 12
    }
    
    factor = ann_factors[freq]

    if modeltype in ['HAR', 'GARCH']:
        rv = np.exp(log_rv_df)
        return np.sqrt(rv) * np.sqrt(factor) * 100
    elif modeltype == 'RFSV':
        return np.exp(log_rv_df) * np.sqrt(factor) * 100
    else:
        raise ValueError(f"Unsupported model type: {modeltype}. Supported types are: HAR, RFSV, GARCH.")


def calculate_comprehensive_metrics(forecasts_dict, actuals_dict, tickers):
    """Calculate performance metrics (RMSE) for all tickers and models"""
    results = {}
    
    for ticker in tickers:
        if ticker not in actuals_dict:
            continue
            
        ticker_actuals = actuals_dict[ticker]
        ticker_forecasts = {}
        
        # Collect forecasts for this ticker
        for model_name in forecasts_dict.keys():
            if ticker in forecasts_dict[model_name]:
                forecast_series = forecasts_dict[model_name][ticker]
                # Align indices
                if len(forecast_series) == len(ticker_actuals):
                    ticker_forecasts[model_name] = forecast_series.values
        
        if not ticker_forecasts:
            continue
            
        # Calculate metrics for this ticker
        ticker_results = {}
        for model, forecast_vals in ticker_forecasts.items():
            try:
                rmse = np.sqrt(mean_squared_error(ticker_actuals.values, forecast_vals))
                ticker_results[model] = {'RMSE': rmse} # Only RMSE
            except Exception as e:
                print(f"Error calculating metrics for {ticker} {model}: {e}")
                ticker_results[model] = {'RMSE': np.nan} # Only RMSE
        
        results[ticker] = ticker_results
    
    return results


def create_summary_table(results, freq_name):
    """Create summary statistics (RMSE) across all tickers"""
    summary_data = []
    
    models = set()
    for ticker_results in results.values():
        models.update(ticker_results.keys())
    
    for model in models:
        rmse_values = []
        
        for ticker, ticker_results in results.items():
            if model in ticker_results:
                rmse_val = ticker_results[model]['RMSE']
                if not (np.isnan(rmse_val)):
                    rmse_values.append(rmse_val)
        
        if rmse_values:
            summary_data.append({
                'Model': model,
                'RMSE_Mean': np.mean(rmse_values),
                'RMSE_Std': np.std(rmse_values),
                'RMSE_Median': np.median(rmse_values),
                'N_Tickers': len(rmse_values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{freq_name} Summary (across {len(results)} tickers):")
    print("="*50)
    print(summary_df.round(4))
    
    return summary_df

def create_performance_heatmap(results, metric='RMSE', freq_name=''):
    """Create heatmap showing performance across tickers and models"""
    # Prepare data for heatmap
    models = set()
    for ticker_results in results.values():
        models.update(ticker_results.keys())
    models = sorted(list(models))
    
    heatmap_data = []
    ticker_names = []
    
    for ticker in sorted(results.keys()):
        ticker_row = []
        for model in models:
            if model in results[ticker]:
                value = results[ticker][model][metric]
                # Replace NaN or None with 0 for visualization
                ticker_row.append(value if not (np.isnan(value) or value is None) else 0)
            else:
                ticker_row.append(0)
        heatmap_data.append(ticker_row)
        ticker_names.append(ticker)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    heatmap_df = pd.DataFrame(heatmap_data, index=ticker_names, columns=models)
    
    # Use different color scales for different metrics
    cmap = 'Reds' # Only RMSE, so a single cmap is fine
    fmt = '.3f'
    
    # Create a mask for zero values (missing data)
    mask = heatmap_df == 0
    
    sns.heatmap(heatmap_df, annot=True, fmt=fmt, cmap=cmap, mask=mask,
                cbar_kws={'label': metric}, linewidths=0.5)
    
    plt.title(f'{freq_name} {metric} Performance Across Tickers and Models', fontsize=14, pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Tickers', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()



def create_model_comparison_plot(summary_df, freq_name):
    """Create bar plots comparing model performance (RMSE only)"""
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6)) # Only one subplot for RMSE
    
    # RMSE comparison
    models = summary_df['Model']
    rmse_means = summary_df['RMSE_Mean']
    rmse_stds = summary_df['RMSE_Std']
    
    bars1 = ax1.bar(models, rmse_means, yerr=rmse_stds, capsize=5, 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_title(f'{freq_name} RMSE Comparison (Mean ± Std)', fontsize=14)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars1, rmse_means, rmse_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def get_model_output_type(model_name):
    model_name = model_name.upper()
    if model_name == 'GARCH':
        return 'GARCH'
    elif model_name == 'HAR':
        return 'HAR'
    elif model_name == 'RFSV':
        return 'RFSV'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
