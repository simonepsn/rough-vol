import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error

from src.analysis_visualization import (
    plot_forecast_comparison,
    calculate_comprehensive_metrics,
    simple_evaluate_single_ticker,
    create_summary_table,
    vol_scaler,
    create_performance_heatmap,
    create_model_comparison_plot
)

# ==============================================================================
#                     --- LOAD ALL TICKER FORECASTS ---
# ==============================================================================

print("=== COMPREHENSIVE MULTI-TICKER ANALYSIS ===")
print("Loading all ticker forecasts...")

try:
    # Load daily data
    with open("forecast_results/all_forecasts_daily.pkl", "rb") as f:
        daily_data = pickle.load(f)

    # Load hourly data  
    with open("forecast_results/all_forecasts_hourly.pkl", "rb") as f:
        hourly_data = pickle.load(f)

    # Load 5-minute data
    with open("forecast_results/all_forecasts_5min.pkl", "rb") as f:
        fivemin_data = pickle.load(f)

    print("✅ Multi-ticker data loaded successfully from pickle files")
    tickers = daily_data['tickers']
    print(f"Analyzing {len(tickers)} tickers: {tickers}")

except FileNotFoundError as e:
    print(f"❌ Error loading pickle files: {e}")
    print("Make sure to run the notebook first to generate the forecast data!")
    exit(1)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ==============================================================================
#                     --- COMPREHENSIVE PERFORMANCE ANALYSIS ---
# ==============================================================================

# Daily analysis
daily_forecasts = {
    'GARCH': daily_data['forecast_garch_d'],
    'HAR': daily_data['forecast_har_d'],
    'RFSV': daily_data['forecast_rfsv_d']
}
daily_results = calculate_comprehensive_metrics(daily_forecasts, daily_data['actuals_d'], tickers)

# Hourly analysis
hourly_forecasts = {
    'GARCH': hourly_data['forecast_garch_h'],
    'HAR': hourly_data['forecast_har_h'], 
    'RFSV': hourly_data['forecast_rfsv_h']
}
hourly_results = calculate_comprehensive_metrics(hourly_forecasts, hourly_data['actuals_h'], tickers)

# 5-minute analysis
fivemin_forecasts = {
    'HAR': fivemin_data['forecast_har_5m'],
    'RFSV': fivemin_data['forecast_rfsv_5m']
}
fivemin_results = calculate_comprehensive_metrics(fivemin_forecasts, fivemin_data['actuals_5m'], tickers)


# ==============================================================================
#                     --- SUMMARY STATISTICS AND RANKINGS ---
# ==============================================================================

daily_summary = create_summary_table(daily_results, "DAILY")
hourly_summary = create_summary_table(hourly_results, "HOURLY") 
fivemin_summary = create_summary_table(fivemin_results, "5-MINUTE")


# ==============================================================================
#                     --- VISUALIZATION: PERFORMANCE HEATMAPS ---
# ==============================================================================

# Create directories for plots
Path("forecast_results/plots/multi_ticker").mkdir(parents=True, exist_ok=True)
Path("forecast_results/plots/individual_tickers").mkdir(parents=True, exist_ok=True)

print("\n" + "="*50)
print("CREATING AGGREGATE VISUALIZATIONS")
print("="*50)

# Create performance heatmaps
fig_daily_rmse = create_performance_heatmap(daily_results, 'RMSE', 'Daily')
fig_daily_rmse.savefig("forecast_results/plots/multi_ticker/daily_rmse_heatmap.png", dpi=300, bbox_inches='tight')

fig_hourly_rmse = create_performance_heatmap(hourly_results, 'RMSE', 'Hourly')
fig_hourly_rmse.savefig("forecast_results/plots/multi_ticker/hourly_rmse_heatmap.png", dpi=300, bbox_inches='tight')

fig_5min_rmse = create_performance_heatmap(fivemin_results, 'RMSE', '5-Minute')
fig_5min_rmse.savefig("forecast_results/plots/multi_ticker/5min_rmse_heatmap.png", dpi=300, bbox_inches='tight')

# Create model comparison plots
fig_daily_comp = create_model_comparison_plot(daily_summary, 'Daily')
fig_daily_comp.savefig("forecast_results/plots/multi_ticker/daily_model_comparison.png", dpi=300, bbox_inches='tight')

fig_hourly_comp = create_model_comparison_plot(hourly_summary, 'Hourly')
fig_hourly_comp.savefig("forecast_results/plots/multi_ticker/hourly_model_comparison.png", dpi=300, bbox_inches='tight')

fig_5min_comp = create_model_comparison_plot(fivemin_summary, '5-Minute')
fig_5min_comp.savefig("forecast_results/plots/multi_ticker/5min_model_comparison.png", dpi=300, bbox_inches='tight')

plt.close('all')
# ==============================================================================
#                     --- INDIVIDUAL TICKER ANALYSIS ---
# ==============================================================================

print("\n" + "="*50)
print("INDIVIDUAL TICKER ANALYSIS")
print("="*50)

# Summary storage for all tickers
all_ticker_summaries = {'daily': {}, 'hourly': {}, '5min': {}}

for ticker in tickers:
    print(f"\n--- Analyzing {ticker} ---")
    
    # ==============================================================================
    #                              --- DAILY ANALYSIS ---
    # ==============================================================================
    
    if ticker in daily_data['actuals_d']:
        actuals_d = daily_data['actuals_d'][ticker]
        forecasts_d_dict = {}
        
        if ticker in daily_data['forecast_garch_d']:
            forecasts_d_dict['GARCH'] = daily_data['forecast_garch_d'][ticker]
        if ticker in daily_data['forecast_har_d']:
            forecasts_d_dict['HAR'] = daily_data['forecast_har_d'][ticker]
        if ticker in daily_data['forecast_rfsv_d']:
            forecasts_d_dict['RFSV'] = daily_data['forecast_rfsv_d'][ticker]
        
        if forecasts_d_dict:
            # Evaluate performance
            results_d = simple_evaluate_single_ticker(actuals_d, forecasts_d_dict)
            all_ticker_summaries['daily'][ticker] = results_d
            
            # Scale for visualization
            actuals_d_scaled = vol_scaler(actuals_d, freq='D')
            forecasts_d_scaled = {}
            for model, forecast in forecasts_d_dict.items():
                forecasts_d_scaled[model] = vol_scaler(forecast, freq='D')
            
            # Create plots
            fig_d = plot_forecast_comparison(actuals_d_scaled, forecasts_d_scaled, freq='D')
            fig_d.suptitle(f'{ticker} - Daily Volatility Forecasts', fontsize=16)
            fig_d.savefig(f"forecast_results/plots/individual_tickers/{ticker}_daily_forecast.png", 
                         dpi=300, bbox_inches='tight')
            plt.close(fig_d)
            
            print(f"  Daily {ticker} - RMSE: {dict(results_d['RMSE'])}")
    
    # ==============================================================================
    #                             --- HOURLY ANALYSIS ---
    # ==============================================================================
    
    if ticker in hourly_data['actuals_h']:
        actuals_h = hourly_data['actuals_h'][ticker]
        forecasts_h_dict = {}
        
        if ticker in hourly_data['forecast_garch_h']:
            forecasts_h_dict['GARCH'] = hourly_data['forecast_garch_h'][ticker]
        if ticker in hourly_data['forecast_har_h']:
            forecasts_h_dict['HAR'] = hourly_data['forecast_har_h'][ticker]
        if ticker in hourly_data['forecast_rfsv_h']:
            forecasts_h_dict['RFSV'] = hourly_data['forecast_rfsv_h'][ticker]
        
        if forecasts_h_dict:
            # Evaluate performance
            results_h = simple_evaluate_single_ticker(actuals_h, forecasts_h_dict)
            all_ticker_summaries['hourly'][ticker] = results_h
            
            # Scale for visualization
            actuals_h_scaled = vol_scaler(actuals_h, freq='h')
            forecasts_h_scaled = {}
            for model, forecast in forecasts_h_dict.items():
                forecasts_h_scaled[model] = vol_scaler(forecast, freq='h')
            
            # Create plots
            fig_h = plot_forecast_comparison(actuals_h_scaled, forecasts_h_scaled, freq='h')
            fig_h.suptitle(f'{ticker} - Hourly Volatility Forecasts', fontsize=16)
            fig_h.savefig(f"forecast_results/plots/individual_tickers/{ticker}_hourly_forecast.png", 
                         dpi=300, bbox_inches='tight')
            plt.close(fig_h)
            
            print(f"  Hourly {ticker} - RMSE: {dict(results_h['RMSE'])}")
    
    # ==============================================================================
    #                            --- 5-MINUTE ANALYSIS ---
    # ==============================================================================
    
    if ticker in fivemin_data['actuals_5m']:
        actuals_5m = fivemin_data['actuals_5m'][ticker]
        forecasts_5m_dict = {}
        
        if ticker in fivemin_data['forecast_har_5m']:
            forecasts_5m_dict['HAR'] = fivemin_data['forecast_har_5m'][ticker]
        if ticker in fivemin_data['forecast_rfsv_5m']:
            forecasts_5m_dict['RFSV'] = fivemin_data['forecast_rfsv_5m'][ticker]
        
        if forecasts_5m_dict:
            # Evaluate performance
            results_5m = simple_evaluate_single_ticker(actuals_5m, forecasts_5m_dict)
            all_ticker_summaries['5min'][ticker] = results_5m
            
            # Scale for visualization
            actuals_5m_scaled = vol_scaler(actuals_5m, freq='5T')
            forecasts_5m_scaled = {}
            for model, forecast in forecasts_5m_dict.items():
                forecasts_5m_scaled[model] = vol_scaler(forecast, freq='5T')
            
            # Create plots
            fig_5m = plot_forecast_comparison(actuals_5m_scaled, forecasts_5m_scaled, freq='5T')
            fig_5m.suptitle(f'{ticker} - 5-Minute Volatility Forecasts', fontsize=16)
            fig_5m.savefig(f"forecast_results/plots/individual_tickers/{ticker}_5min_forecast.png", 
                          dpi=300, bbox_inches='tight')
            plt.close(fig_5m)
            
            print(f"  5-min {ticker} - RMSE: {dict(results_5m['RMSE'])}")

# ==============================================================================
#                         --- AGGREGATE RESULTS SUMMARY ---
# ==============================================================================

print("\n" + "="*60)
print("INDIVIDUAL TICKER ANALYSIS SUMMARY")
print("="*60)

# Create aggregate summary tables
for freq_name, freq_summaries in all_ticker_summaries.items():
    if freq_summaries:
        print(f"\n{freq_name.upper()} FREQUENCY - Individual Ticker Performance:")
        print("-" * 60)
        
        # Combine all ticker results into one table
        combined_results = []
        for ticker, results_df in freq_summaries.items():
            for model in results_df.index:
                combined_results.append({
                    'Ticker': ticker,
                    'Model': model,
                    'RMSE': results_df.loc[model, 'RMSE'] # Only RMSE
                })
        
        if combined_results:
            summary_df = pd.DataFrame(combined_results)
            
            # Show top performers for each model
            print("\nTop 3 performers by RMSE for each model:")
            for model in summary_df['Model'].unique():
                model_data = summary_df[summary_df['Model'] == model].nsmallest(3, 'RMSE')
                print(f"  {model}:")
                for _, row in model_data.iterrows():
                    print(f"    {row['Ticker']}: RMSE={row['RMSE']:.4f}") # Only RMSE
            
            # Save detailed results
            summary_df.to_csv(f"forecast_results/individual_ticker_results_{freq_name}.csv", index=False)


# ==============================================================================
#                                  --- SAVE RESULTS ---
# ==============================================================================

print("\n" + "="*50)
print("SAVING COMPREHENSIVE ANALYSIS RESULTS")
print("="*50)

# Save summary statistics
daily_summary.to_csv("forecast_results/daily_multi_ticker_summary.csv", index=False)
hourly_summary.to_csv("forecast_results/hourly_multi_ticker_summary.csv", index=False)  
fivemin_summary.to_csv("forecast_results/5min_multi_ticker_summary.csv", index=False)

print("✅ Comprehensive analysis completed successfully!")
print(f"✅ Results analyzed for {len(tickers)} tickers across 3 frequencies")
print("✅ Aggregate visualizations saved in forecast_results/plots/multi_ticker/")
print("✅ Individual ticker plots saved in forecast_results/plots/individual_tickers/")
print("✅ Summary statistics saved in forecast_results/*_summary.csv")
print("✅ Individual ticker results saved in forecast_results/individual_ticker_results_*.csv")


# ==============================================================================
#                                  --- THANKS ---
# ==============================================================================