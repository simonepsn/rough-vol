import pandas as pd
import numpy as np
import warnings
import pickle
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(os.path.abspath("/home/simonepsn/Desktop/rough-vol/")) 

from statsmodels.tsa.stattools import adfuller

from src.data_preparation import calculate_log_rv, prepare_har_data
from src.garch import forecast_garch_rolling
from src.har import forecast_har_rolling
from src.rfsv import rolling_forecast_rfsv





# ==============================================================================
#                               --- SETUP ---
# ==============================================================================

warnings.filterwarnings("ignore")

raw_dir = 'other/data/raw_data'
output_path = 'other/data/df.csv'
holdout_period = 25

# Initialize list to collect all model coefficients
all_coeffs = []



# ==============================================================================
#                   --- DATA LOADING AND PREPARATION ---
# ==============================================================================

print("Loading multi-ticker data...")
df_5m = pd.read_csv('other/data/raw_data/5min_data.csv', sep=';', index_col=0, parse_dates=True)
df_h = pd.read_csv('other/data/raw_data/1h_data.csv', sep=';', index_col=0, parse_dates=True)
df_d = pd.read_csv('other/data/raw_data/1d_data.csv', sep=';', index_col=0, parse_dates=True)

print(f"Data shapes - Daily: {df_d.shape}, Hourly: {df_h.shape}, 5min: {df_5m.shape}")
print(f"Available tickers: {[col.replace('_close', '') for col in df_d.columns if '_close' in col]}")

# Calculate log returns for each ticker (for GARCH)
print("\nCalculating log returns for all tickers...")

lret_daily_data = {}
lret_hourly_data = {}
lret_5min_data = {}

for col in df_d.columns:
    if '_close' in col:
        ticker = col.replace('_close', '')
        
        # Calculate log returns for each frequency
        lret_daily_data[ticker] = np.log(df_d[col] / df_d[col].shift(1)).dropna()
        lret_hourly_data[ticker] = np.log(df_h[col] / df_h[col].shift(1)).dropna()
        lret_5min_data[ticker] = np.log(df_5m[col] / df_5m[col].shift(1)).dropna()

all_tickers = [col.replace('_close', '') for col in df_d.columns if '_close' in col]

freq_data_mapping = {
    'daily': (df_d, 'Daily'),
    'hourly': (df_h, 'Hourly'), 
    '5min': (df_5m, '5-Minute')
}

ticker_groups = [
    all_tickers[:4],
    all_tickers[4:7],
    all_tickers[7:10]
]

tickers_per_figure = 3
num_figures = (len(all_tickers) + tickers_per_figure - 1) // tickers_per_figure

for freq_name, (df_freq, freq_label) in freq_data_mapping.items():
    for group_idx, group_tickers in enumerate(ticker_groups):
        fig, axes = plt.subplots(1, len(group_tickers), figsize=(5*len(group_tickers), 4))
        if len(group_tickers) == 1:
            axes = [axes]
        fig.suptitle(f'{freq_label} Price Series - Group {group_idx + 1}', fontsize=16, fontweight='bold')
        for ticker_idx, ticker in enumerate(group_tickers):
            col_name = f'{ticker}_close'
            ax = axes[ticker_idx]
            if col_name in df_freq.columns:
                ax.plot(df_freq.index, df_freq[col_name], color='blue', alpha=0.7, linewidth=1)
                ax.set_title(f'{ticker}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.set_xlabel('Date', fontsize=9)
                ax.set_ylabel('Price', fontsize=9)
            else:
                ax.axis('off')
        plt.tight_layout()
        filename = f'other/figures/price_series_{freq_name}_group_{group_idx + 1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# Convert to DataFrames
lret_d = pd.DataFrame(lret_daily_data)
lret_h = pd.DataFrame(lret_hourly_data)
lret_5m = pd.DataFrame(lret_5min_data)

print(f"Log returns calculated for {len(lret_d.columns)} tickers")

# Calculate log realized volatility for each ticker (for RFSV/HAR)
print("\nCalculating log realized volatility for all tickers...")
lrv_d = calculate_log_rv(df_d, resample_freq='1D')
lrv_h = calculate_log_rv(df_h, resample_freq='1h') 
lrv_5m = calculate_log_rv(df_5m, resample_freq='5min')

print(f"Log RV shapes - Daily: {lrv_d.shape}, Hourly: {lrv_h.shape}, 5min: {lrv_5m.shape}")

# Prepare HAR data for each ticker
print("\nPreparing HAR data for all tickers...")
har_d_data = prepare_har_data(lrv_d, freq='D')
har_h_data = prepare_har_data(lrv_h, freq='h')
har_5m_data = prepare_har_data(lrv_5m, freq='5min')

print(f"HAR data prepared for {len(har_d_data)} tickers (daily)")

# ==============================================================================#
#                              --- ADF TEST ---                                 #
# ==============================================================================#

adf_results_d = {}
adf_results_h = {}
adf_results_5m = {}

# ADF test sui log realized volatility (per testare persistenza della volatilità)
for ticker in lrv_d.columns:
    # Daily log RV
    adf_result = adfuller(lrv_d[ticker].dropna())
    adf_results_d[ticker] = {
        'adf_stat': adf_result[0],
        'p_value': adf_result[1],
    }

for ticker in lrv_h.columns:
    # Hourly log RV
    adf_result = adfuller(lrv_h[ticker].dropna())
    adf_results_h[ticker] = {
        'adf_stat': adf_result[0],
        'p_value': adf_result[1],
    }

for ticker in lrv_5m.columns:
    # 5-minute log RV
    adf_result = adfuller(lrv_5m[ticker].dropna())
    adf_results_5m[ticker] = {
        'adf_stat': adf_result[0],
        'p_value': adf_result[1],
    }

# Generate a summary table to print as a png
freq = ['daily', 'hourly', '5min']
colors = {
    'daily': '#1f77b4',
    'hourly': '#ff7f0e',
    '5min': '#2ca02c'
}

data = []
for freq_name in freq:
    for ticker in lrv_d.columns:
        if freq_name == 'daily':
            adf_stat = adf_results_d[ticker]['adf_stat']
            p_value = adf_results_d[ticker]['p_value']
        elif freq_name == 'hourly':
            adf_stat = adf_results_h[ticker]['adf_stat']
            p_value = adf_results_h[ticker]['p_value']
        elif freq_name == '5min':
            adf_stat = adf_results_5m[ticker]['adf_stat']
            p_value = adf_results_5m[ticker]['p_value']
        
        data.append({
            'freq': freq_name,
            'color': colors[freq_name],
            'ticker': ticker,
            'adf_stat': adf_stat,
            'p_value': p_value
        })

df_adf = pd.DataFrame(data)

# Create a pivot table for better visualization

pivot_adf = df_adf.pivot_table(
    index='ticker',
    columns='freq',
    values=['adf_stat', 'p_value'],
    aggfunc='first'
).reset_index()
pivot_adf.columns = ['_'.join(col).strip() for col in pivot_adf.columns.values]
pivot_adf.rename(columns={'ticker_': 'ticker'}, inplace=True)

# Create the table visualization
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Format the data for better readability
pivot_display = pivot_adf.copy()

# Round numerical values e usa notazione scientifica per p-values piccoli
numeric_cols = [col for col in pivot_display.columns if col != 'ticker']
for col in numeric_cols:
    if 'adf_stat' in col:
        pivot_display[col] = pivot_display[col].round(3)
    elif 'p_value' in col:
        # Usa sempre notazione scientifica per p-values
        pivot_display[col] = pivot_display[col].apply(lambda x: f"{x:.2e}" if not pd.isna(x) else 'NaN')

# Create the table
table = ax.table(cellText=pivot_display.values, 
                colLabels=pivot_display.columns,
                cellLoc='center', 
                loc='center',
                bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2)

# Color code the header
for i in range(len(pivot_display.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code p-values based on significance
for i in range(1, len(pivot_display) + 1):
    for j in range(len(pivot_display.columns)):
        col_name = pivot_display.columns[j]
        if 'p_value' in col_name:
            p_val_original = pivot_adf.iloc[i-1, j]
            if p_val_original < 0.01:
                table[(i, j)].set_facecolor('#90EE90')  # Light green for highly significant
            elif p_val_original < 0.05:
                table[(i, j)].set_facecolor('#FFFF99')  # Light yellow for significant
            elif p_val_original < 0.10:
                table[(i, j)].set_facecolor('#FFB366')  # Light orange for marginally significant
            else:
                table[(i, j)].set_facecolor('#FFB3B3')  # Light red for non-significant 

# Create legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='p < 0.01 (Highly Significant)'),
    plt.Rectangle((0,0),1,1, facecolor='#FFFF99', label='p < 0.05 (Significant)'),
    plt.Rectangle((0,0),1,1, facecolor='#FFB366', label='p < 0.10 (Marginally Significant)'),
    plt.Rectangle((0,0),1,1, facecolor='#FFB3B3', label='p ≥ 0.10 (Non-Significant)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# Save as PNG
plt.tight_layout()
plt.savefig('forecast_results/adf_test_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ ADF test results table saved as PNG: forecast_results/adf_test_results.png")

# Also save as CSV for reference
pivot_adf.to_csv('forecast_results/adf_test_results.csv', index=False)
print("✅ ADF test results also saved as CSV: forecast_results/adf_test_results.csv")

# Determine available analyses based on data length
analysis_frequencies = []
window_sizes = {}

# Daily data

train_lret_d = lret_d.iloc[:-holdout_period]
train_lrv_d = lrv_d.iloc[:-holdout_period]

train_har_d = {
    ticker: df.iloc[:-holdout_period].copy()
    for ticker, df in har_d_data.items()
}

actuals_d = lrv_d.iloc[-holdout_period:]
window_size_d = min(len(train_lrv_d) // 2, 250)
analysis_frequencies.append('daily')
window_sizes['daily'] = window_size_d
print(f"Daily analysis enabled: {len(train_lrv_d)} training obs, {len(actuals_d)} test obs")

# Hourly data
train_lret_h = lret_h.iloc[:-holdout_period]
train_lrv_h = lrv_h.iloc[:-holdout_period]

train_har_h = {
    ticker: df.iloc[:-holdout_period].copy()
    for ticker, df in har_h_data.items()
}

actuals_h = lrv_h.iloc[-holdout_period:]
window_size_h = min(len(train_lrv_h) // 2, 250)
analysis_frequencies.append('hourly')
window_sizes['hourly'] = window_size_h
print(f"Hourly analysis enabled: {len(train_lrv_h)} training obs, {len(actuals_h)} test obs")

# 5-minute data
train_lret_5m = lret_5m.iloc[:-holdout_period]
train_lrv_5m = lrv_5m.iloc[:-holdout_period]

train_har_5m = {
    ticker: df.iloc[:-holdout_period].copy()
    for ticker, df in har_5m_data.items()
}

actuals_5m = lrv_5m.iloc[-holdout_period:]
window_size_5m = min(len(train_lrv_5m) // 2, 250)
analysis_frequencies.append('5min')
window_sizes['5min'] = window_size_5m
print(f"5-minute analysis enabled: {len(train_lrv_5m)} training obs, {len(actuals_5m)} test obs")

print(f"Enabled analyses: {analysis_frequencies}")

# Data validation
print("\n--- Data Validation ---")
for freq in analysis_frequencies:
    if freq == 'daily':
        print(f"Daily - Train period: {train_lrv_d.index.min()} to {train_lrv_d.index.max()}")
        print(f"Daily - Test period: {actuals_d.index.min()} to {actuals_d.index.max()}")
    elif freq == 'hourly':
        print(f"Hourly - Train period: {train_lrv_h.index.min()} to {train_lrv_h.index.max()}")
        print(f"Hourly - Test period: {actuals_h.index.min()} to {actuals_h.index.max()}")
    elif freq == '5min':
        print(f"5min - Train period: {train_lrv_5m.index.min()} to {train_lrv_5m.index.max()}")
        print(f"5min - Test period: {actuals_5m.index.min()} to {actuals_5m.index.max()}")

# ==============================================================================
#                              --- DAILY ANALYSIS ---
# ==============================================================================

if 'daily' in analysis_frequencies:
    print("\n" + "="*50)
    print("DAILY ANALYSIS")
    print("="*50)
    
    # Initialize forecast dictionaries for each model
    forecast_garch_d = {}
    forecast_har_d = {}
    forecast_rfsv_d = {}

    # Get all tickers
    tickers = list(lrv_d.columns)
    print(f"Processing {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        
        # Get ticker-specific data
        ticker_lret_d = train_lret_d[ticker]
        ticker_lrv_d = train_lrv_d[ticker]
        ticker_har_d = train_har_d[ticker]
        ticker_actuals_d = actuals_d[ticker]
        
        # GARCH
        try:
            forecast_garch, garch_coefs = forecast_garch_rolling(
                ticker_lret_d, 
                horizon=holdout_period, 
                window_size=window_sizes['daily'], 
                last_log_rv=ticker_lrv_d.iloc[-1]
            )

            forecast_garch_d[ticker] = forecast_garch.set_axis(ticker_actuals_d.index)
            
            # Collect coefficients
            for _, row in garch_coefs.iterrows():
                for param in ['omega', 'alpha', 'beta']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'GARCH',
                        'Frequency': 'daily',
                        'Param': param,
                        'Value': row[param],
                        'Date': row.get('date', None),
                        'Step': row.get('step', None)
                    })

        except Exception as e:
            print(f"GARCH error for {ticker}: {e}")
        
        # HAR
        try:
            forecast_har, har_coefs = forecast_har_rolling(
                ticker_har_d, 
                horizon=holdout_period, 
                window_size=window_sizes['daily'], 
                last_log_rv=ticker_lrv_d.iloc[-1]
            )

            forecast_har_d[ticker] = forecast_har.set_axis(ticker_actuals_d.index)
            
            # Collect coefficients
            for _, row in har_coefs.iterrows():
                for param in ['const', 'daily_lag', 'weekly_lag', 'monthly_lag']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'HAR',
                        'Frequency': 'daily',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': row.get('date', None),
                        'Step': row.get('step', None)
                    })

        except Exception as e:
            print(f"HAR error for {ticker}: {e}")
        
        # RFSV
        try:
            scales_d = [1, 2, 5, 10]
            forecast, rfsv_coefs = rolling_forecast_rfsv(
                ticker_lrv_d, 
                scales=scales_d, 
                horizon=holdout_period, 
                rolling_window=100, 
                n_sims=5, 
                freq='D'
            )
        
            forecast_rfsv_d[ticker] = forecast.set_axis(ticker_actuals_d.index)
            
            # Collect coefficients
            for _, row in rfsv_coefs.iterrows():
                for param in ['H', 'nu', 'n_points']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'RFSV',
                        'Frequency': 'daily',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': None,
                        'Step': None
                    })

        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")
    
    # Create summary DataFrames (using first ticker as representative)
    forecast_df_d = pd.DataFrame()

    for ticker in tickers:
        for model_name, model_dict in zip(['GARCH', 'HAR', 'RFSV'], [forecast_garch_d, forecast_har_d, forecast_rfsv_d]):
            col_name = f"{ticker}_{model_name}"
            forecast_df_d[col_name] = model_dict[ticker].reset_index(drop=True)

    print(f"Daily forecasting completed: {len(forecast_df_d)} predictions for {len(tickers)} tickers")
else:
    print("Skipping daily analysis due to insufficient data")
    forecasts_d_dict = {}
    forecast_df_d = pd.DataFrame()

# ==============================================================================
#                           --- 1-HOUR ANALYSIS ---
# ==============================================================================

if 'hourly' in analysis_frequencies:
    print("\n" + "="*50)
    print("HOURLY ANALYSIS")
    print("="*50)
    
    # Initialize forecast dictionaries for each model
    forecast_garch_h = {}
    forecast_har_h = {}
    forecast_rfsv_h = {}

    # Get all tickers
    tickers = list(lrv_h.columns)
    print(f"Processing {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        
        # Get ticker-specific data
        ticker_lret_h = train_lret_h[ticker]
        ticker_lrv_h = train_lrv_h[ticker]
        ticker_har_h = train_har_h[ticker]
        ticker_actuals_h = actuals_h[ticker]
        
        # GARCH
        try:
            forecast_garch, garch_coefs = forecast_garch_rolling(
                ticker_lret_h, 
                horizon=holdout_period, 
                window_size=window_sizes['hourly'], 
                last_log_rv=ticker_lrv_h.iloc[-1]
            )

            forecast_garch_h[ticker] = forecast_garch.set_axis(ticker_actuals_h.index)
            
            # Collect coefficients
            for _, row in garch_coefs.iterrows():
                for param in ['omega', 'alpha', 'beta']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'GARCH',
                        'Frequency': 'hourly',
                        'Param': param,
                        'Value': row[param],
                        'Date': row.get('date', None),
                        'Step': row.get('step', None)
                    })

        except Exception as e:
            print(f"GARCH error for {ticker}: {e}")

        # HAR
        try:
            forecast_har, har_coefs = forecast_har_rolling(
                ticker_har_h, 
                horizon=holdout_period, 
                window_size=window_sizes['hourly'], 
                last_log_rv=ticker_lrv_h.iloc[-1]
            )

            forecast_har_h[ticker] = forecast_har.set_axis(ticker_actuals_h.index)
            
            # Collect coefficients
            for _, row in har_coefs.iterrows():
                for param in ['const', 'daily_lag', 'weekly_lag', 'monthly_lag']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'HAR',
                        'Frequency': 'hourly',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': row.get('date', None),
                        'Step': row.get('step', None)
                    })

        except Exception as e:
            print(f"HAR error for {ticker}: {e}")       

        # RFSV
        try:
            scales_h = [1, 2, 5, 10, 20, 50]
            forecast, rfsv_coefs = rolling_forecast_rfsv(
                ticker_lrv_h, 
                scales=scales_h, 
                horizon=holdout_period, 
                rolling_window=window_sizes['hourly'], 
                n_sims=5, 
                freq='h'
            )

            forecast_rfsv_h[ticker] = forecast.set_axis(ticker_actuals_h.index)
            
            # Collect coefficients
            for _, row in rfsv_coefs.iterrows():
                for param in ['H', 'nu', 'n_points']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'RFSV',
                        'Frequency': 'hourly',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': None,
                        'Step': None
                    })

        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")

    forecast_df_h = pd.DataFrame()

    for ticker in tickers:
        for model_name, model_dict in zip(['GARCH', 'HAR', 'RFSV'], [forecast_garch_h, forecast_har_h, forecast_rfsv_h]):
            col_name = f"{ticker}_{model_name}"
            forecast_df_h[col_name] = model_dict[ticker].reset_index(drop=True)

    print(f"Hourly forecasting completed: {len(forecast_df_h)} predictions for {len(tickers)} tickers")
else:
    print("Skipping hourly analysis due to insufficient data")
    forecasts_h_dict = {}
    forecast_df_h = pd.DataFrame()
# ==============================================================================
#                          --- 5-MINUTES ANALYSIS ---
# ==============================================================================

if '5min' in analysis_frequencies:
    print("\n" + "="*50)
    print("5-MINUTE ANALYSIS")
    print("="*50)
    
    # Initialize forecast dictionaries for each model
    # forecast_garch_5m = {}
    forecast_har_5m = {}
    forecast_rfsv_5m = {}

    # Get all tickers
    tickers = list(lrv_5m.columns)
    print(f"Processing {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        
        # Get ticker-specific data
        ticker_lret_5m = train_lret_5m[ticker] if ticker in lret_5m.columns else None
        ticker_lrv_5m = train_lrv_5m[ticker]
        ticker_har_5m = train_har_5m[ticker]
        ticker_actuals_5m = actuals_5m[ticker]

        # GARCH (commented because GARCH is not implemented for 5-minute data for statistical reasons)
        #
        # try:
        #     forecast_garch = forecast_garch_rolling(
        #         ticker_lret_5m, 
        #         horizon=holdout_period, 
        #         window_size=window_sizes['5min'], 
        #         last_log_rv=ticker_lrv_5m.iloc[-1]
        #     )
        #
        #     forecast_garch_5m[ticker] = forecast_garch.set_axis(ticker_actuals_5m.index)
        #
        # except Exception as e:
        #     print(f"GARCH error for {ticker}: {e}")
        
        # HAR
        try:
            forecast_har, har_coefs = forecast_har_rolling(
                ticker_har_5m, 
                horizon=holdout_period, 
                window_size=window_sizes['5min'], 
                last_log_rv=ticker_lrv_5m.iloc[-1]
            )

            forecast_har_5m[ticker] = forecast_har.set_axis(ticker_actuals_5m.index)
            
            # Collect coefficients
            for _, row in har_coefs.iterrows():
                for param in ['const', 'daily_lag', 'weekly_lag', 'monthly_lag']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'HAR',
                        'Frequency': '5min',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': row.get('date', None),
                        'Step': row.get('step', None)
                    })

        except Exception as e:
            print(f"HAR error for {ticker}: {e}")
        
        # RFSV
        try:
            scales_5m = [1, 2, 5, 10, 20, 50, 100]
            forecast, rfsv_coefs = rolling_forecast_rfsv(
                ticker_lrv_5m, 
                scales=scales_5m, 
                horizon=holdout_period, 
                rolling_window=window_sizes['5min'], 
                n_sims=5, 
                freq='5min'
            )

            forecast_rfsv_5m[ticker] = forecast.set_axis(ticker_actuals_5m.index)
            
            # Collect coefficients
            for _, row in rfsv_coefs.iterrows():
                for param in ['H', 'nu', 'n_points']:
                    all_coeffs.append({
                        'Ticker': ticker,
                        'Model': 'RFSV',
                        'Frequency': '5min',
                        'Param': param,
                        'Value': row.get(param, None),
                        'Date': None,
                        'Step': None
                    })

        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")

    # Create summary DataFrame - only using HAR and RFSV for 5-minute (GARCH skipped)
    forecast_df_5m = pd.DataFrame()

    for ticker in tickers:
        # HAR forecasts
        col_name_har = f"{ticker}_HAR"
        forecast_df_5m[col_name_har] = forecast_har_5m[ticker].reset_index(drop=True)
        
        # RFSV forecasts
        col_name_rfsv = f"{ticker}_RFSV"
        forecast_df_5m[col_name_rfsv] = forecast_rfsv_5m[ticker].reset_index(drop=True)

    print(f"5-minute forecasting completed: {len(forecast_df_5m)} predictions for {len(tickers)} tickers")
else:
    print("Skipping 5-minute analysis due to insufficient data")
    forecast_har_5m = {}
    forecast_rfsv_5m = {}
    forecast_df_5m = pd.DataFrame()


# ==============================================================================
#                     --- SAVE ALL TICKER FORECASTS FOR ANALYSIS ---
# ==============================================================================

print("\n" + "="*50)
print("SAVING ALL TICKER FORECASTS")
print("="*50)


# Create results directory
os.makedirs("forecast_results", exist_ok=True)

# Save all ticker forecasts and actuals for comprehensive analysis
if 'daily' in analysis_frequencies:
    with open("forecast_results/all_forecasts_daily.pkl", "wb") as f:
        pickle.dump({
            'forecast_garch_d': forecast_garch_d,
            'forecast_har_d': forecast_har_d, 
            'forecast_rfsv_d': forecast_rfsv_d,
            'actuals_d': actuals_d,
            'tickers': list(lrv_d.columns)
        }, f)
    print("All daily ticker forecasts saved")

if 'hourly' in analysis_frequencies:
    with open("forecast_results/all_forecasts_hourly.pkl", "wb") as f:
        pickle.dump({
            'forecast_garch_h': forecast_garch_h,
            'forecast_har_h': forecast_har_h,
            'forecast_rfsv_h': forecast_rfsv_h, 
            'actuals_h': actuals_h,
            'tickers': list(lrv_h.columns)
        }, f)
    print("All hourly ticker forecasts saved")

if '5min' in analysis_frequencies:
    with open("forecast_results/all_forecasts_5min.pkl", "wb") as f:
        pickle.dump({
            'forecast_har_5m': forecast_har_5m,
            'forecast_rfsv_5m': forecast_rfsv_5m,
            'actuals_5m': actuals_5m,
            'tickers': list(lrv_5m.columns)
        }, f)
    print("All 5-minute ticker forecasts saved")

# Save all model coefficients
coeffs_df = pd.DataFrame(all_coeffs)
coeffs_df.to_csv('forecast_results/all_model_coefficients.csv', index=False)
print("✅ All model coefficients saved to forecast_results/all_model_coefficients.csv")

# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================