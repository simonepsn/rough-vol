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

def calculate_log_rv(df, resample_freq='1D', ticker_col=None):
    """
    Calculate log realized variance using intraday returns for multiple tickers.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index and ticker columns
        resample_freq (str): Frequency for resampling ('1D', '1h', '5min')
        ticker_col (str): If specified, calculate for single ticker only
    
    Returns:
        pd.DataFrame or pd.Series: Log realized variance for all tickers or single ticker
    """
    epsilon = 1e-10
    
    if ticker_col is not None:
        # Single ticker calculation
        if ticker_col not in df.columns:
            raise ValueError(f"Column {ticker_col} not found in DataFrame")
        
        price_series = pd.to_numeric(df[ticker_col], errors='coerce').dropna()
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        squared_log_returns = log_returns ** 2
        realized_variance = squared_log_returns.resample(resample_freq).sum()
        log_realized_variance = np.log(realized_variance + epsilon)
        
        return log_realized_variance
    
    else:
        # Multi-ticker calculation
        log_rv_dict = {}
        
        for col in df.columns:
            try:
                price_series = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(price_series) < 10:  # Skip if insufficient data
                    print(f"Skipping {col}: insufficient data ({len(price_series)} observations)")
                    continue
                
                # Calculate log returns
                log_returns = np.log(price_series / price_series.shift(1)).dropna()
                
                # Calculate squared log returns
                squared_log_returns = log_returns ** 2
                
                # Resample to desired frequency and sum to get realized variance
                realized_variance = squared_log_returns.resample(resample_freq).sum()
                
                # Take log of realized variance
                log_realized_variance = np.log(realized_variance + epsilon)
                
                # Extract ticker name from column (remove '_close' suffix)
                ticker_name = col.replace('_close', '') if '_close' in col else col
                log_rv_dict[ticker_name] = log_realized_variance
                
                print(f"Calculated log RV for {ticker_name}: {len(log_realized_variance)} observations")
                
            except Exception as e:
                print(f"Error processing {col}: {e}")
                continue
        
        if not log_rv_dict:
            raise ValueError("No valid ticker data found")
        
        # Return DataFrame with tickers as columns
        log_rv_df = pd.DataFrame(log_rv_dict)
        return log_rv_df


def prepare_har_data(log_rv_data, freq='D', ticker=None):
    """
    Prepare HAR DataFrame for different frequencies and multiple tickers.
    
    Args:
        log_rv_data (pd.DataFrame or pd.Series): Log RV data for multiple tickers or single series
        freq (str): Frequency ('D', 'h', '5min')
        ticker (str): If specified, prepare data for single ticker only
    
    Returns:
        dict or pd.DataFrame: HAR data for all tickers (dict) or single ticker (DataFrame)
    """
    if freq == 'D':
        periods_in_day = 1
    elif freq == 'h':
        periods_in_day = 1 * 8
    elif freq == '5min':
        periods_in_day = 8 * 12
    else:
        raise ValueError("Freq not supported. Use 'D', 'h', or '5min'.")
    
    weekly_window = periods_in_day * 5
    monthly_window = periods_in_day * 22
    
    def prepare_single_ticker_har(log_rv_series, ticker_name):
        """Helper function to prepare HAR data for a single ticker"""
        df = pd.DataFrame({'log_rv': log_rv_series})
        
        df['daily_lag'] = df['log_rv'].shift(1)
        df['weekly_lag'] = df['log_rv'].rolling(window=weekly_window).mean().shift(1)
        df['monthly_lag'] = df['log_rv'].rolling(window=monthly_window).mean().shift(1)
        
        df.dropna(inplace=True)
        
        print(f"HAR data prepared for {ticker_name}: {len(df)} observations")
        return df
    
    # Handle single ticker case
    if ticker is not None:
        if isinstance(log_rv_data, pd.DataFrame):
            if ticker not in log_rv_data.columns:
                raise ValueError(f"Ticker {ticker} not found in log_rv_data columns")
            return prepare_single_ticker_har(log_rv_data[ticker], ticker)
        else:
            # Assume it's a Series for the specified ticker
            return prepare_single_ticker_har(log_rv_data, ticker)
    
    # Handle multiple tickers case
    if isinstance(log_rv_data, pd.Series):
        # Single series, treat as one ticker
        return prepare_single_ticker_har(log_rv_data, 'default')
    
    # Multiple tickers case
    har_data_dict = {}
    
    for ticker_name in log_rv_data.columns:
        try:
            har_data = prepare_single_ticker_har(log_rv_data[ticker_name], ticker_name)
            har_data_dict[ticker_name] = har_data
        except Exception as e:
            print(f"Error preparing HAR data for {ticker_name}: {e}")
            continue
    
    if not har_data_dict:
        raise ValueError("No valid HAR data could be prepared")
    
    return har_data_dict
