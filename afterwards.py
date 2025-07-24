import numpy as np
import pandas as pd

from src.analysis_visualization import (
    plot_forecast_comparison, 
    plot_metrics_comparison, 
    evaluate_forecasts,
    plot_error_diagnostics,
    vol_scaler
)

# ==============================================================================
#                                --- SETUP ---
# ==============================================================================

forecast_df_d = pd.read_csv("forecast_results/forecast_d.csv", sep=";", index_col=0, parse_dates=True)
forecast_df_h = pd.read_csv("forecast_results/forecast_h.csv", sep=";", index_col=0, parse_dates=True)
forecast_df_5m = pd.read_csv("forecast_results/forecast_5m.csv", sep=";", index_col=0, parse_dates=True)

actuals_d = forecast_df_d['actuals']
actuals_h = forecast_df_h['actuals']
actuals_5m = forecast_df_5m['actuals']

# ==============================================================================
#                          --- DATA PREPARATION ---
# ==============================================================================
forecast_df_h = forecast_df_h.groupby(forecast_df_h.index.date).sum()
forecast_df_h.index = pd.to_datetime(forecast_df_h.index)
forecast_df_5m = forecast_df_5m.groupby(forecast_df_5m.index.date).sum()
forecast_df_5m.index = pd.to_datetime(forecast_df_5m.index)
actuals_h = actuals_h.groupby(actuals_h.index.date).sum()
actuals_h.index = pd.to_datetime(actuals_h.index)
actuals_5m = actuals_5m.groupby(actuals_5m.index.date).sum()
actuals_5m.index = pd.to_datetime(actuals_5m.index)

# Scale the forecasts correctly
forecast_garch_d_scaled = vol_scaler(forecast_df_d['GARCH'], freq='D')
forecast_har_d_scaled = vol_scaler(forecast_df_d['HAR'], freq='D')
forecast_rfsv_d_scaled = vol_scaler(forecast_df_d['RFSV'], freq='D')

forecast_garch_h_scaled = vol_scaler(forecast_df_h['GARCH'], freq='h')
forecast_har_h_scaled = vol_scaler(forecast_df_h['HAR'], freq='h')
forecast_rfsv_h_scaled = vol_scaler(forecast_df_h['RFSV'], freq='h')

# forecast_garch_5m_scaled = vol_scaler(forecast_garch_5m, freq='5m')
forecast_har_5m_scaled = vol_scaler(forecast_df_5m['HAR'], freq='5m')
forecast_rfsv_5m_scaled = vol_scaler(forecast_df_5m['RFSV'], freq='5m')

actuals_d_scaled = vol_scaler(actuals_d, freq='D')
actuals_h_scaled = vol_scaler(actuals_h, freq='h')
actuals_5m_scaled = vol_scaler(actuals_5m, freq='5m')


forecasts_d_dict = {
    'GARCH': forecast_garch_d_scaled,
    'HAR': forecast_har_d_scaled,
    'RFSV': forecast_rfsv_d_scaled
}

forecasts_h_dict = {
    'GARCH': forecast_garch_h_scaled,
    'HAR': forecast_har_h_scaled,
    'RFSV': forecast_rfsv_h_scaled
}

forecasts_5m_dict = {
    'HAR': forecast_har_5m_scaled,
    'RFSV': forecast_rfsv_5m_scaled
}

results_summary_d, rmse_d, qlike_d = evaluate_forecasts(actuals_d_scaled, forecasts_d_dict)
results_summary_h, rmse_h, qlike_h = evaluate_forecasts(actuals_h_scaled, forecasts_h_dict)
results_summary_5m, rmse_5m, qlike_5m = evaluate_forecasts(actuals_5m_scaled, forecasts_5m_dict)

# ==============================================================================
#                               --- EVALUATION ---
# ==============================================================================

# Daily Results

fig_d = plot_forecast_comparison(actuals_d_scaled, forecasts_d_dict, freq='Daily')
fig_d.savefig("forecast_results/plots/daily_forecast_comparison.png", dpi=500)

fig_metrics_d = plot_metrics_comparison(rmse_d, qlike_d, freq='Daily')
fig_metrics_d.savefig("forecast_results/plots/daily_metrics_comparison.png", dpi=500)

fig_error_d = plot_error_diagnostics(actuals_d_scaled, forecasts_d_dict, freq='Daily')
fig_error_d.savefig("forecast_results/plots/daily_metrics_errors.png", dpi=500)

# Hourly Results

fig_h = plot_forecast_comparison(actuals_h_scaled, forecasts_h_dict, freq='Hourly')
fig_h.savefig("forecast_results/plots/hourly_forecast_comparison.png", dpi=500)

fig_metrics_h = plot_metrics_comparison(rmse_h, qlike_h, freq='Hourly')
fig_metrics_h.savefig("forecast_results/plots/hourly_metrics_comparison.png", dpi=500)

fig_error_h = plot_error_diagnostics(actuals_h_scaled, forecasts_h_dict, freq='Hourly')
fig_error_h.savefig("forecast_results/plots/hourly_metrics_errors.png", dpi=500)

# 5-Minute Results

fig_5m = plot_forecast_comparison(actuals_5m_scaled, forecasts_5m_dict, freq='5minutes')
fig_5m.savefig("forecast_results/plots/5minutes_forecast_comparison.png", dpi=500)

fig_metrics_5m = plot_metrics_comparison(rmse_5m, qlike_5m, freq='5minutes')
fig_metrics_5m.savefig("forecast_results/plots/5minutes_metrics_comparison.png", dpi=500)

fig_error_5m = plot_error_diagnostics(actuals_5m_scaled, forecasts_5m_dict, freq='5minutes')
fig_error_5m.savefig("forecast_results/plots/5minutes_metrics_errors.png", dpi=500)

print("Analysis and visualizations completed successfully!")

# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================