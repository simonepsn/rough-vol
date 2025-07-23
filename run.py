import numpy as np
import pandas as pd

from src.data_preparation import load_and_clean, calculate_log_rv, prepare_har_data
from src.garch import estimate_garch, forecast_garch, forecast_garch_rolling
from src.har import forecast_har_rolling
from src.rfsv import estimate_h_loglog, build_fbm_covariance_matrix ,forecast_RFSV
from src.analysis_visualization import (
    qlike_loss, 
    evaluate_forecasts, 
    plot_forecast_comparison, 
    plot_metrics_comparison, 
    plot_error_analysis
)


# ==============================================================================
#                               --- SETUP ---
# ==============================================================================

raw_dir = 'other/data/raw_data'
output_path = 'other/data/df.csv'
holdout_days = 25



# ==============================================================================
#                   --- DATA LOADING AND PREPARATION ---
# ==============================================================================

df = load_and_clean(raw_data_directory=raw_dir, file_pattern='SPX*.csv', output_path=output_path)

# GARCH inputs
price_d = df['price'].resample('D').last()
lret_d = np.log(price_d).diff().dropna()
price_h = df['price'].resample('h').last()
lret_h = np.log(price_h).diff().dropna()
price_5m = df['price'].resample('5min').last()
lret_5m = np.log(price_5m).diff().dropna()

# RFSV/HAR inputs
lrv_d = calculate_log_rv(df, price_col='price', resample_freq='1D').squeeze()
lrv_h = calculate_log_rv(df, price_col='price', resample_freq='1h').squeeze()
lrv_5m = calculate_log_rv(df, price_col='price', resample_freq='5min').squeeze()

# HAR inputs
har_d_data = prepare_har_data(lrv_d, freq='D')
har_h_data = prepare_har_data(lrv_h, freq='h')
har_5m_data = prepare_har_data(lrv_5m, freq='5min')



# SPLIT DATA INTO TRAINING AND HOLDOUT (TEST) SETS

# Daily
train_lret_d = lret_d[:-holdout_days]
train_lrv_d = lrv_d[:-holdout_days]
actuals_d = lrv_d[-holdout_days:]

# 1-Hour
holdout_h = holdout_days * 24
train_lret_h = lret_h[:-holdout_h]
train_lrv_h = lrv_h[:-holdout_h]
actuals_h = lrv_h[-holdout_h:]

# 5-minutes
holdout_5m = holdout_days * 288
train_lret_5m = lret_5m[:-holdout_5m]
train_lrv_5m = lrv_5m[:-holdout_5m]
actuals_5m = lrv_5m[-holdout_5m:]



# ==============================================================================
#                              --- DAILY ANALYSIS ---
# ==============================================================================

# GARCH
forecast_garch_d = forecast_garch_rolling(lret_d, horizon=holdout_days, window_size=252, last_log_rv=train_lrv_d.iloc[-1])

# HAR
forecast_har_d = forecast_har_rolling(har_d_data, horizon=holdout_days, window_size=252, last_log_rv=train_lrv_d.iloc[-1])

# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_d, _ = estimate_h_loglog(train_lrv_d, scales, q=1)
nu_est_d = train_lrv_d.diff().dropna().std()
forecast_rfsv_d = forecast_RFSV(train_lrv_d, h_est_d, nu_est_d, horizon=holdout_days, freq='D', truncation_window=252, n_sims=5, use_last_value=True, index=actuals_d.index)

# Ensure forecasts have the same index as actuals
forecast_garch_d.index = actuals_d.index
forecast_har_d.index = actuals_d.index
forecast_rfsv_d.index = actuals_d.index

# Store forecasts in a DataFrame
forecasts_d_dict = {'GARCH': forecast_garch_d, 'HAR': forecast_har_d, 'RFSV': forecast_rfsv_d}
forecast_df_d = pd.DataFrame(forecasts_d_dict)

# ==============================================================================
#                           --- 1-HOUR ANALYSIS ---
# ==============================================================================

# GARCH
forecast_garch_h = forecast_garch_rolling(lret_h, horizon=holdout_days * 24, window_size=252, last_log_rv=train_lrv_h.iloc[-1])

# HAR
forecast_har_h = forecast_har_rolling(har_h_data, horizon=holdout_days * 24, window_size=252, last_log_rv=train_lrv_h.iloc[-1])


# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_h, _ = estimate_h_loglog(train_lrv_h, scales, q=1)
nu_est_h = train_lrv_h.diff().dropna().std()
forecast_rfsv_h = forecast_RFSV(train_lrv_h, h_est_h, nu_est_h, horizon=holdout_h, freq='h', truncation_window=252, n_sims=5, use_last_value=True, index=actuals_h.index)

# Keep the index consistent with actuals
forecast_garch_h.index = actuals_h.index
forecast_har_h.index = actuals_h.index
forecast_rfsv_h.index = actuals_h.index

# Collect 1-hour forecasts
forecasts_h_dict = {'GARCH': forecast_garch_h, 'HAR': forecast_har_h, 'RFSV': forecast_rfsv_h}
forecast_df_h = pd.DataFrame(forecasts_h_dict)



# ==============================================================================
#                          --- 5-MINUTES ANALYSIS ---
# ==============================================================================

# GARCH
# forecast_garch_5m = forecast_garch_rolling(lret_5m, horizon=holdout_days * 24 * 12, window_size=252, last_log_rv=train_lrv_5m.iloc[-1])
# Note: GARCH is not expected to perform well at this high frequency due to non-trading periods and intraday seasonality, but is included for completeness.

# HAR
forecast_har_5m = forecast_har_rolling(har_5m_data, horizon=holdout_days * 24 * 12, window_size=252, last_log_rv=train_lrv_5m.iloc[-1])


# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_5m, _ = estimate_h_loglog(train_lrv_5m, scales, q=1)
nu_est_5m = train_lrv_5m.diff().dropna().std()
forecast_rfsv_5m = forecast_RFSV(train_lrv_5m, h_est_5m, nu_est_5m, horizon=holdout_5m, freq='5T', truncation_window=252, n_sims=5, use_last_value=True, index=actuals_5m.index)

# Keep the index consistent with actuals
# forecast_garch_5m.index = actuals_5m.index
forecast_har_5m.index = actuals_5m.index
forecast_rfsv_5m.index = actuals_5m.index


# Collect 5-minutes forecasts
forecasts_5m_dict = {'HAR': forecast_har_5m, 'RFSV': forecast_rfsv_5m}
forecast_df_5m = pd.DataFrame(forecasts_5m_dict)

# For the sake of computationally doable things (and statistical usefulness) we leave this here {'GARCH': forecast_garch_5m}

# ==============================================================================
#       --- PERFORMANCE EVALUATION & VISUALIZATION (RUNS AT THE END) ---
# ==============================================================================

# --- Daily Results ---
results_summary_d, rmse_d, qlike_d = evaluate_forecasts(actuals_d, forecasts_d_dict)
results_summary_d.to_csv("forecast_results/results_summary_d.csv", sep=";")

# --- Hourly Results ---
results_summary_h, rmse_h, qlike_h = evaluate_forecasts(actuals_h, forecasts_h_dict)
results_summary_h.to_csv("forecast_results/results_summary_h.csv", sep=";")


# --- 5-Minute Results ---
results_summary_5m, rmse_5m, qlike_5m = evaluate_forecasts(actuals_5m, forecasts_5m_dict)
results_summary_5m.to_csv("forecast_results/results_summary_5m.csv", sep=";")


# ==============================================================================
#                           --- SAVE FINAL RESULTS ---
# ==============================================================================

forecast_df_d['actuals'] = actuals_d
forecast_df_h['actuals'] = actuals_h
forecast_df_5m['actuals'] = actuals_5m

forecast_df_d.to_csv("forecast_results/forecast_d.csv", sep=";")
forecast_df_h.to_csv("forecast_results/forecast_h.csv", sep=";")
forecast_df_5m.to_csv("forecast_results/forecast_5m.csv", sep=";")

print("Finished!")

# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================