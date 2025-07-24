import pandas as pd
import numpy as np
import glob
import os

def load_and_clean(raw_data_directory, file_pattern, output_path):
    """
    Loads and merges CSVs, cleanes and stores them in dir.

    Args:
        raw_data_directory (str): Raw CSVs dir.
        file_pattern (str): How to find CSVs.
        output_path (str): Where to store clean CSVs.

    Returns:
        pd.DataFrame: Clean DF ready for analysis.
    """
    all_files = glob.glob(os.path.join(raw_data_directory, file_pattern))
    all_files.sort()
    
    if not all_files:
        print(f"No file with pattern:'{file_pattern}' in '{raw_data_directory}'")
        return pd.DataFrame()

    columns = ['date & time', 'open', 'high', 'low', 'close', 'price']
    
    df_list = [pd.read_csv(f, delimiter=";", names=columns) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # ---  Data Cleaning ---

    df['date & time'] = pd.to_datetime(df['date & time'], format='%Y%m%d %H%M%S')
    df = df.drop(columns=['open', 'high', 'low', 'price'])
    df = df.rename(columns={'date & time': 'datetime', 'close': 'price'})
    df = df.set_index('datetime').sort_index()
    df.dropna(inplace=True)

    df.to_csv(output_path, sep=',', index=True, header=True)
        
    return df

def calculate_log_rv(df, price_col='price', resample_freq='1D'):
    """
    Compute log realized volatility (lrv) from HF data.
    Enhanced filtering for high-frequency data to remove non-trading periods.

    Args:
        df (pd.DataFrame): DF.
        price_col (str): price column.
        resample_freq (str): Resampling frequency (es. '1D', '4H', '1H').

    Returns:
        pd.Series: pd.series log realized volatility with timestamp.
    """

    log_returns = np.log(df[price_col]).diff().dropna()

    # For intraday frequencies, filter trading hours before computing RV
    if resample_freq in ['5min', '5T', '1H', 'h']:
        # Keep only weekdays
        log_returns = log_returns[log_returns.index.weekday < 5]
        
        # Filter trading hours (9:30 AM - 4:00 PM EST)
        if resample_freq in ['5min', '5T']:
            log_returns = log_returns.between_time('09:30', '16:00')
        elif resample_freq in ['1H', 'h']:
            log_returns = log_returns.between_time('09:00', '16:00')
        
        print(f"After trading hours filter: {len(log_returns)} observations")

    squared_log_returns = log_returns**2

    realized_variance = squared_log_returns.resample(resample_freq).sum()
    
    # Enhanced filtering for high-frequency data
    if resample_freq in ['5min', '5T']:
        # For 5-minute data, use more aggressive filtering
        min_threshold = realized_variance.quantile(0.05)  # Bottom 5%
        max_threshold = realized_variance.quantile(0.95)  # Top 5%
        realized_variance = realized_variance[
            (realized_variance > min_threshold) & 
            (realized_variance < max_threshold)
        ]
        
        # Additional filter: remove periods with too few observations per day
        daily_counts = realized_variance.groupby(realized_variance.index.date).count()
        valid_dates = daily_counts[daily_counts >= 50].index  # At least 50 obs per day
        date_filter = pd.Series(realized_variance.index.date).isin(valid_dates)
        realized_variance = realized_variance[date_filter.values]
        
    elif resample_freq in ['1H', 'h']:
        # For hourly data
        min_threshold = realized_variance.quantile(0.02)  # Bottom 2%
        max_threshold = realized_variance.quantile(0.98)  # Top 2%
        realized_variance = realized_variance[
            (realized_variance > min_threshold) & 
            (realized_variance < max_threshold)
        ]
        
        # Additional filter: remove periods with too few observations per day
        daily_counts = realized_variance.groupby(realized_variance.index.date).count()
        valid_dates = daily_counts[daily_counts >= 5].index  # At least 5 obs per day
        date_filter = pd.Series(realized_variance.index.date).isin(valid_dates)
        realized_variance = realized_variance[date_filter.values]
        
    else:
        # Daily data - original filtering
        min_threshold = realized_variance.quantile(0.01)  # Bottom 1%
        realized_variance = realized_variance[realized_variance > min_threshold]
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    realized_variance = np.maximum(realized_variance, epsilon)
    
    log_realized_variance = np.log(realized_variance)
    
    # Remove extreme outliers
    log_realized_variance = log_realized_variance.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Frequency-specific outlier filtering
    if resample_freq in ['5min', '5T']:
        # Tighter bounds for 5-minute data
        Q1 = log_realized_variance.quantile(0.10)
        Q3 = log_realized_variance.quantile(0.90)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
    elif resample_freq in ['1H', 'h']:
        # Moderate bounds for hourly data
        Q1 = log_realized_variance.quantile(0.05)
        Q3 = log_realized_variance.quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
    else:
        # Original bounds for daily data
        Q1 = log_realized_variance.quantile(0.25)
        Q3 = log_realized_variance.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
    
    log_realized_variance = log_realized_variance[
        (log_realized_variance >= lower_bound) & 
        (log_realized_variance <= upper_bound)
    ]
    
    print(f"Filtered log-RV for {resample_freq}: {len(log_realized_variance)} observations, range [{log_realized_variance.min():.3f}, {log_realized_variance.max():.3f}]")
    
    return log_realized_variance


def prepare_har_data(log_rv_series, freq='D'):
    """
    Prepare HAR df for different frequencies.
    """

    if freq == 'D':
        periods_in_day = 1
    elif freq == 'h':
        periods_in_day = 1 * 24
    elif freq == '5min':
        periods_in_day = 24 * 12
    else:
        raise ValueError("Freq not supported. Use 'D', 'h', or '5min'.")
    
    weekly_window = periods_in_day * 5
    monthly_window = periods_in_day * 22
    
    df = pd.DataFrame({'log_rv': log_rv_series})
    
    df['daily_lag'] = df['log_rv'].shift(1)
    df['weekly_lag'] = df['log_rv'].rolling(window=weekly_window).mean().shift(1)
    df['monthly_lag'] = df['log_rv'].rolling(window=monthly_window).mean().shift(1)
    
    df.dropna(inplace=True)

    return df
