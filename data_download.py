import yfinance as yf
import pandas as pd

ticker = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "IBM"]

df_5m = yf.download(ticker, period='60d', interval='5m', auto_adjust=True)
df_h = yf.download(ticker, period='60d', interval='1h', auto_adjust=True)
df_d = yf.download(ticker, period='2y', interval='1d', auto_adjust=True)

def remove_timezone(df):
    if df.index.tz is not None:  # se l'indice ha timezone
        return df.tz_convert(None)
    return df  # già tz-naive


df_5m = remove_timezone(df_5m['Close'])
df_h  = remove_timezone(df_h['Close'])
df_d  = remove_timezone(df_d['Close'])

df_5m = df_5m.rename(columns=lambda x: f"{x}_close")
df_h  = df_h.rename(columns=lambda x: f"{x}_close")
df_d  = df_d.rename(columns=lambda x: f"{x}_close")

print(df_5m.head())

# Save to CSV
df_5m.to_csv('other/data/raw_data/5min_data.csv', sep=';', index=True, header=True)
df_h.to_csv('other/data/raw_data/1h_data.csv', sep=';', index=True, header=True)
df_d.to_csv('other/data/raw_data/1d_data.csv', sep=';', index=True, header=True)