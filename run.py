import pandas as pd
import numpy as np
import warnings
import pickle
import sys
import os

os.chdir(os.path.abspath("/home/simonepsn/Desktop/rough-vol/")) 

from src.data_preparation import load_and_clean, calculate_log_rv, prepare_har_data
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

# Determine available analyses based on data length
analysis_frequencies = []
window_sizes = {}

# Daily data

train_lret_d = lret_d.iloc[:-holdout_period]
train_lrv_d = lrv_d.iloc[:-holdout_period]
actuals_d = lrv_d.iloc[-holdout_period:]
window_size_d = min(len(train_lrv_d) // 2, 252)
analysis_frequencies.append('daily')
window_sizes['daily'] = window_size_d
print(f"Daily analysis enabled: {len(train_lrv_d)} training obs, {len(actuals_d)} test obs")

# Hourly data
train_lret_h = lret_h.iloc[:-holdout_period]
train_lrv_h = lrv_h.iloc[:-holdout_period]
actuals_h = lrv_h.iloc[-holdout_period:]
window_size_h = min(len(train_lrv_h) // 2, 252)
analysis_frequencies.append('hourly')
window_sizes['hourly'] = window_size_h
print(f"Hourly analysis enabled: {len(train_lrv_h)} training obs, {len(actuals_h)} test obs")

# 5-minute data
train_lret_5m = lret_5m.iloc[:-holdout_period]
train_lrv_5m = lrv_5m.iloc[:-holdout_period]
actuals_5m = lrv_5m.iloc[-holdout_period:]
window_size_5m = min(len(train_lrv_5m) // 2, 252)
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
        ticker_lret_d = lret_d[ticker]
        ticker_lrv_d = train_lrv_d[ticker]
        ticker_har_d = har_d_data[ticker]
        ticker_actuals_d = actuals_d[ticker]
        
        # GARCH
        try:
            forecast_garch_d[ticker] = forecast_garch_rolling(
                ticker_lret_d, 
                horizon=holdout_period, 
                window_size=window_sizes['daily'], 
                last_log_rv=ticker_lrv_d.iloc[-1]
            )
        except Exception as e:
            print(f"GARCH error for {ticker}: {e}")
            forecast_garch_d[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_d.index)
        
        # HAR
        try:
            forecast_har_d[ticker] = forecast_har_rolling(
                ticker_har_d, 
                horizon=holdout_period, 
                window_size=window_sizes['daily'], 
                last_log_rv=ticker_lrv_d.iloc[-1]
            )
        except Exception as e:
            print(f"HAR error for {ticker}: {e}")
            forecast_har_d[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_d.index)
        
        # RFSV
        try:
            scales_d = [1, 2, 4, 8, 16, 22]
            forecast_rfsv_d[ticker] = rolling_forecast_rfsv(
                ticker_lrv_d, 
                scales=scales_d, 
                horizon=holdout_period, 
                rolling_window=window_sizes['daily'], 
                n_sims=5, 
                freq='D'
            )
        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")
            forecast_rfsv_d[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_d.index)
    
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
        ticker_lret_h = lret_h[ticker]
        ticker_lrv_h = train_lrv_h[ticker]
        ticker_har_h = har_h_data[ticker]
        ticker_actuals_h = actuals_h[ticker]
        
        # GARCH
        try:
            forecast_garch_h[ticker] = forecast_garch_rolling(
                ticker_lret_h, 
                horizon=holdout_period, 
                window_size=window_sizes['hourly'], 
                last_log_rv=ticker_lrv_h.iloc[-1]
            )
        except Exception as e:
            print(f"GARCH error for {ticker}: {e}")
            forecast_garch_h[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_h.index)
        
        # HAR
        try:
            forecast_har_h[ticker] = forecast_har_rolling(
                ticker_har_h, 
                horizon=holdout_period, 
                window_size=window_sizes['hourly'], 
                last_log_rv=ticker_lrv_h.iloc[-1]
            )
        except Exception as e:
            print(f"HAR error for {ticker}: {e}")
            forecast_har_h[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_h.index)
        
        # RFSV
        try:
            scales_h = [1, 2, 4, 8, 16, 32, 48]
            forecast_rfsv_h[ticker] = rolling_forecast_rfsv(
                ticker_lrv_h, 
                scales=scales_h, 
                horizon=holdout_period, 
                rolling_window=window_sizes['hourly'], 
                n_sims=5, 
                freq='h'
            )
        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")
            forecast_rfsv_h[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_h.index)
       
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
    forecast_garch_5m = {}
    forecast_har_5m = {}
    forecast_rfsv_5m = {}
    
    # Get all tickers
    tickers = list(lrv_5m.columns)
    print(f"Processing {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        
        # Get ticker-specific data
        ticker_lret_5m = lret_5m[ticker] if ticker in lret_5m.columns else None
        ticker_lrv_5m = train_lrv_5m[ticker]
        ticker_har_5m = har_5m_data[ticker]
        ticker_actuals_5m = actuals_5m[ticker]
        
        # GARCH (commented out for computational efficiency)
        # try:
        #     forecast_garch_5m[ticker] = forecast_garch_rolling(
        #         ticker_lret_5m, 
        #         horizon=holdout_period, 
        #         window_size=window_sizes['5min'], 
        #         last_log_rv=ticker_lrv_5m.iloc[-1]
        #     )
        # except Exception as e:
        #     print(f"GARCH error for {ticker}: {e}")
        #     forecast_garch_5m[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_5m.index)
        
        # HAR
        try:
            forecast_har_5m[ticker] = forecast_har_rolling(
                ticker_har_5m, 
                horizon=holdout_period, 
                window_size=window_sizes['5min'], 
                last_log_rv=ticker_lrv_5m.iloc[-1]
            )
        except Exception as e:
            print(f"HAR error for {ticker}: {e}")
            forecast_har_5m[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_5m.index)
        
        # RFSV
        try:
            scales_5m = [1, 2, 4, 8, 16, 32, 64, 128]
            forecast_rfsv_5m[ticker] = rolling_forecast_rfsv(
                ticker_lrv_5m, 
                scales=scales_5m, 
                horizon=holdout_period, 
                rolling_window=window_sizes['5min'], 
                n_sims=5, 
                freq='5min'
            )
        except Exception as e:
            print(f"RFSV error for {ticker}: {e}")
            forecast_rfsv_5m[ticker] = pd.Series([np.nan] * holdout_period, index=ticker_actuals_5m.index)
    
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

# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================