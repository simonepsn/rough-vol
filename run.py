import numpy as np
import pandas as pd

from src.data_preparation import load_and_clean, calculate_log_rv, prepare_har_data
from src.garch import estimate_garch, forecast_garch
from src.har import estimate_har, forecast_har_iterative
from src.rfsv import estimate_h_loglog, build_fbm_covariance_matrix ,forecast_RFSV
from src.analysis_visualization import qlike_loss, evaluate_forecasts, plot_forecast_comparison, plot_metrics_comparison


# --- SETUP ---

raw_dir = 'other/data/raw_data'
output_path = 'other/data/df.csv'
holdout_days = 15 



# --- DATA LOADING AND PREPARATION ---

df = load_and_clean(raw_data_directory=raw_dir, file_pattern='SPX*.csv', output_path=output_path)

# GARCH inputs
price_d = df['price'].resample('D').last()
lret_d = np.log(price_d).diff().dropna()
price_h = df['price'].resample('H').last()
lret_h = np.log(price_h).diff().dropna()
price_5m = df['price'].resample('5m').last()
lret_5m = np.log(price_5m).diff().dropna()

# RFSV/HAR inputs
lrv_d = calculate_log_rv(df, price_col=['price'], resample_freq='1D').squeeze()
lrv_h = calculate_log_rv(df, price_col=['price'], resample_freq='1h').squeeze()
lrv_5m = calculate_log_rv(df, price_col=['price'], resample_freq='5min').squeeze()

# HAR inputs
har_d_data = prepare_har_data(lrv_d, freq='D')
har_h_data = prepare_har_data(lrv_h, freq='H')
har_5m_data = prepare_har_data(lrv_5m, freq='5min')



# --- SPLIT DATA INTO TRAINING AND HOLDOUT (TEST) SETS ---

# Daily
train_lret_d = lret_d[:-holdout_days]
train_lrv_d = lrv_d[:-holdout_days]
actuals_d = lrv_d[-holdout_days:]

# Hourly
holdout_h = holdout_days * 24
train_lret_h = lret_h[:-holdout_h]
train_lrv_h = lrv_h[:-holdout_h]
actuals_h = lrv_h[-holdout_h:]

# 5-minute
holdout_5m = holdout_days * 288
train_lret_5m = lret_5m[:-holdout_5m]
train_lrv_5m = lrv_5m[:-holdout_5m]
actuals_5m = lrv_5m[-holdout_5m:]


#
# --- ANALYSIS FOR DAILY FREQUENCY ---
#

# GARCH
garch_d = estimate_garch(train_lret_d)
forecast_garch_d = forecast_garch(garch_d, horizon=holdout_days, last_known_date=train_lret_d.index[-1], freq='D')

# HAR
har_d_data = prepare_har_data(train_lrv_d, freq='D')
har_d = estimate_har(har_d_data)
latest_lags_d = har_d_data.iloc[-1][['daily_lag', 'weekly_lag', 'monthly_lag']]
forecast_har_d = forecast_har_iterative(har_d, latest_lags_d, horizon=holdout_days, last_known_date=har_d_data.index[-1], freq='D')

# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_d, _ = estimate_h_loglog(train_lrv_d, scales, q=1)
nu_est_d = train_lrv_d.diff().dropna().std()
forecast_rfsv_d = forecast_RFSV(train_lrv_d, h_est_d, nu_est_d, horizon=holdout_days, truncation_window=252, n_sims=5)

# --- Performance Evaluation (Daily) ---
forecasts_d_dict = {'GARCH': forecast_garch_d, 'HAR': forecast_har_d, 'RFSV': forecast_rfsv_d}
forecast_df_d = pd.DataFrame(forecasts_d_dict)
results_d = evaluate_forecasts(actuals_d, forecasts_d_dict)
print("\n--- Performance Metrics (Daily) ---")
print(results_d)

# --- Visualization (Daily) ---
plot_forecast_comparison(actuals_d, forecasts_d_dict, freq='Daily')


#
# --- ANALYSIS FOR 1-HOUR FREQUENCY ---
#

# GARCH
garch_h = estimate_garch(train_lret_h)
forecast_garch_h = forecast_garch(garch_h, horizon=holdout_h, last_known_date=train_lret_h.index[-1], freq='h')

# HAR
har_h_data = prepare_har_data(train_lrv_h, freq='by Hour')
har_h = estimate_har(har_h_data)
latest_lags_h = har_h_data.iloc[-1][['daily_lag', 'weekly_lag', 'monthly_lag']]
forecast_har_h = forecast_har_iterative(har_h, latest_lags_h, horizon=holdout_h, last_known_date=har_h_data.index[-1], freq='h')

# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_h, _ = estimate_h_loglog(train_lrv_h, scales, q=1)
nu_est_h = train_lrv_h.diff().dropna().std()
forecast_rfsv_h = forecast_RFSV(train_lrv_h, h_est_h, nu_est_h, horizon=holdout_h, truncation_window=252, n_sims=5)

# --- Performance Evaluation (by Hour) ---
forecasts_h_dict = {'GARCH': forecast_garch_h, 'HAR': forecast_har_h, 'RFSV': forecast_rfsv_h}
forecast_df_h = pd.DataFrame(forecasts_h_dict)
results_h = evaluate_forecasts(actuals_h, forecasts_h_dict)
print("\n--- Performance Metrics (Daily) ---")
print(results_h)

# --- Visualization (by Hour) ---
plot_forecast_comparison(actuals_h, forecasts_h_dict, freq='h')


#
# --- ANALYSIS FOR 5-MINUTE FREQUENCY ---
#

# GARCH
garch_5m = estimate_garch(train_lret_5m)
forecast_garch_5m = forecast_garch(garch_5m, horizon=holdout_5m, last_known_date=train_lret_5m.index[-1], freq='5T')
# Note: GARCH is not expected to perform well at this high frequency due to non-trading periods and intraday seasonality, but is included for completeness.

# HAR
har_5m_data = prepare_har_data(train_lrv_5m, freq='5min')
har_5m = estimate_har(har_5m_data)
latest_lags_5m = har_5m_data.iloc[-1][['daily_lag', 'weekly_lag', 'monthly_lag']]
forecast_har_5m = forecast_har_iterative(har_5m, latest_lags_5m, horizon=holdout_5m, last_known_date=har_5m_data.index[-1], freq='5T')

# RFSV
scales = [1, 2, 5, 10, 20, 50, 100, 200]
h_est_5m, _ = estimate_h_loglog(train_lrv_5m, scales, q=1)
nu_est_5m = train_lrv_5m.diff().dropna().std()
forecast_rfsv_5m = forecast_RFSV(train_lrv_5m, h_est_5m, nu_est_5m, horizon=holdout_5m, truncation_window=252, n_sims=5)

# --- Performance Evaluation (by 5-minutes) ---
forecasts_5m_dict = {'GARCH': forecast_garch_5m, 'HAR': forecast_har_5m, 'RFSV': forecast_rfsv_5m}
forecast_df_5m = pd.DataFrame(forecasts_5m_dict)
results_5m = evaluate_forecasts(actuals_5m, forecasts_5m_dict)
print("\n--- Performance Metrics (5-minute) ---")
print(results_5m)

# --- Visualization (by 5-minutes) ---
plot_horizon = 288
plot_forecast_comparison(actuals_5m.iloc[:plot_horizon], 
                         {model: forecast.iloc[:plot_horizon] for model, forecast in forecasts_5m_dict.items()}, 
                         freq_text='5-minute (First Day of Holdout)')

forecast_df_d.merge(actuals_d, how="inner")
forecast_df_h.merge(actuals_h, how="inner")
forecast_df_5m.merge(actuals_5m, how="inner")

forecast_df_d.to_csv("other/data/forecast_d.csv", sep=";")
forecast_df_h.to_csv("other/data/forecast_h.csv", sep=";")
forecast_df_5m.to_csv("other/data/forecast_5m.csv", sep=";")


#
# --- GRAZIE PER L'ATTENZIONE!
#