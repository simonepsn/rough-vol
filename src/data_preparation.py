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

    Args:
        df (pd.DataFrame): DF.
        price_col (str): price column.
        resample_freq (str): Resampling frequency (es. '1D', '4H', '1H').

    Returns:
        pd.Series: pd.series log realized volatility with timestamp.
    """

    log_returns = np.log(df[price_col]).diff().dropna()

    squared_log_returns = log_returns**2

    realized_variance = squared_log_returns.resample(resample_freq).sum()
    
    log_realized_variance = np.log(realized_variance)
    
    log_realized_variance = log_realized_variance.replace([np.inf, -np.inf], np.nan).dropna()
    
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
