import numpy as np
import pandas as pd
from pathlib import Path

from src.analysis_visualization import (
    plot_forecast_comparison, 
    plot_metrics_comparison,
    plot_error_diagnostics,
    evaluate_forecasts,
    vol_scaler
)

# ==============================================================================
#                           --- LOAD DATA (NO SCALING YET) ---
# ==============================================================================

forecast_df_d = pd.read_csv("forecast_results/forecast_d.csv", sep=";", index_col=0, parse_dates=True)
forecast_df_h = pd.read_csv("forecast_results/forecast_h.csv", sep=";", index_col=0, parse_dates=True)
forecast_df_5m = pd.read_csv("forecast_results/forecast_5m.csv", sep=";", index_col=0, parse_dates=True)

# DEBUG: Check for extreme values
print("=== DEBUGGING DATA RANGES ===")
print(f"Daily actuals range: {forecast_df_d['actuals'].min():.3f} to {forecast_df_d['actuals'].max():.3f}")
print(f"Daily GARCH range: {forecast_df_d['GARCH'].min():.3f} to {forecast_df_d['GARCH'].max():.3f}")
print(f"Daily HAR range: {forecast_df_d['HAR'].min():.3f} to {forecast_df_d['HAR'].max():.3f}")
print(f"Daily RFSV range: {forecast_df_d['RFSV'].min():.3f} to {forecast_df_d['RFSV'].max():.3f}")

print(f"\n5min actuals range: {forecast_df_5m['actuals'].min():.3f} to {forecast_df_5m['actuals'].max():.3f}")
print(f"5min HAR range: {forecast_df_5m['HAR'].min():.3f} to {forecast_df_5m['HAR'].max():.3f}")
print(f"5min RFSV range: {forecast_df_5m['RFSV'].min():.3f} to {forecast_df_5m['RFSV'].max():.3f}")

# Check for NaN or infinite values
print(f"\nDaily NaN count: actuals={forecast_df_d['actuals'].isna().sum()}, HAR={forecast_df_d['HAR'].isna().sum()}, RFSV={forecast_df_d['RFSV'].isna().sum()}")
print(f"5min NaN count: actuals={forecast_df_5m['actuals'].isna().sum()}, HAR={forecast_df_5m['HAR'].isna().sum()}, RFSV={forecast_df_5m['RFSV'].isna().sum()}")
print("==============================\n")

# ==============================================================================
#                           --- EVALUATE ON LOG-RV SCALE FIRST ---
# ==============================================================================

# Daily (log-RV scale)
forecasts_d_dict_log = {
    'GARCH': forecast_df_d['GARCH'],
    'HAR': forecast_df_d['HAR'],
    'RFSV': forecast_df_d['RFSV']
}

# Hourly (log-RV scale)
forecasts_h_dict_log = {
    'GARCH': forecast_df_h['GARCH'],
    'HAR': forecast_df_h['HAR'],
    'RFSV': forecast_df_h['RFSV']
}

# 5-Minute (log-RV scale)
forecasts_5m_dict_log = {
    'HAR': forecast_df_5m['HAR'],
    'RFSV': forecast_df_5m['RFSV']
}

# Evaluate on log-RV scale (no overflow issues) with proper frequency filtering
results_d, rmse_d, qlike_d = evaluate_forecasts(forecast_df_d['actuals'], forecasts_d_dict_log, freq='D')
results_h, rmse_h, qlike_h = evaluate_forecasts(forecast_df_h['actuals'], forecasts_h_dict_log, freq='H')
results_5m, rmse_5m, qlike_5m = evaluate_forecasts(forecast_df_5m['actuals'], forecasts_5m_dict_log, freq='5min')

print("--- Performance Metrics (Log-RV Scale) ---")
print("\nDaily:")
print(results_d)
print("\nHourly:")
print(results_h)
print("\n5-Minute:")
print(results_5m)

# ==============================================================================
#                           --- SCALE FOR VISUALIZATION ONLY ---
# ==============================================================================

# Scale to percentage volatility for visualization
actuals_d_scaled = vol_scaler(forecast_df_d['actuals'], freq='D')
actuals_h_scaled = vol_scaler(forecast_df_h['actuals'], freq='H')  # Fix: 'H' not 'h'
actuals_5m_scaled = vol_scaler(forecast_df_5m['actuals'], freq='5min')

forecasts_d_dict_scaled = {
    'GARCH': vol_scaler(forecast_df_d['GARCH'], freq='D'),
    'HAR': vol_scaler(forecast_df_d['HAR'], freq='D'),
    'RFSV': vol_scaler(forecast_df_d['RFSV'], freq='D')
}

forecasts_h_dict_scaled = {
    'GARCH': vol_scaler(forecast_df_h['GARCH'], freq='H'),  # Fix: 'H' not 'h'
    'HAR': vol_scaler(forecast_df_h['HAR'], freq='H'),
    'RFSV': vol_scaler(forecast_df_h['RFSV'], freq='H')
}

forecasts_5m_dict_scaled = {
    'HAR': vol_scaler(forecast_df_5m['HAR'], freq='5min'),
    'RFSV': vol_scaler(forecast_df_5m['RFSV'], freq='5min')
}

# ==============================================================================
#                           --- VISUALIZATION ---
# ==============================================================================

# Create plots directory
Path("forecast_results/plots").mkdir(parents=True, exist_ok=True)

# Daily plots
fig_d = plot_forecast_comparison(actuals_d_scaled, forecasts_d_dict_scaled, freq='D')
fig_d.savefig("forecast_results/plots/daily_forecast_comparison.png", dpi=300, bbox_inches='tight')

fig_metrics_d = plot_metrics_comparison(rmse_d, qlike_d, freq='Daily')
fig_metrics_d.savefig("forecast_results/plots/daily_metrics_comparison.png", dpi=300, bbox_inches='tight')

fig_error_d = plot_error_diagnostics(actuals_d_scaled, forecasts_d_dict_scaled, freq='D')
fig_error_d.savefig("forecast_results/plots/daily_error_analysis.png", dpi=300, bbox_inches='tight')

# Hourly plots
fig_h = plot_forecast_comparison(actuals_h_scaled, forecasts_h_dict_scaled, freq='H')
fig_h.savefig("forecast_results/plots/hourly_forecast_comparison.png", dpi=300, bbox_inches='tight')

fig_metrics_h = plot_metrics_comparison(rmse_h, qlike_h, freq='Hourly')
fig_metrics_h.savefig("forecast_results/plots/hourly_metrics_comparison.png", dpi=300, bbox_inches='tight')

fig_error_h = plot_error_diagnostics(actuals_h_scaled, forecasts_h_dict_scaled, freq='H')
fig_error_h.savefig("forecast_results/plots/hourly_error_analysis.png", dpi=300, bbox_inches='tight')

# 5-Minute plots
fig_5m = plot_forecast_comparison(actuals_5m_scaled, forecasts_5m_dict_scaled, freq='5min')
fig_5m.savefig("forecast_results/plots/5minutes_forecast_comparison.png", dpi=300, bbox_inches='tight')

fig_metrics_5m = plot_metrics_comparison(rmse_5m, qlike_5m, freq='5-Minutes')
fig_metrics_5m.savefig("forecast_results/plots/5minutes_metrics_comparison.png", dpi=300, bbox_inches='tight')

fig_error_5m = plot_error_diagnostics(actuals_5m_scaled, forecasts_5m_dict_scaled, freq='5min')
fig_error_5m.savefig("forecast_results/plots/5minutes_error_analysis.png", dpi=300, bbox_inches='tight')

print("\nAll plots saved successfully!")

# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================