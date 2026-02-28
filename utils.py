import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime

COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "DOGE-USD": "Dogecoin",
    "SHIB-USD": "Shiba Inu", 
    "FLOKI-USD": "Floki",
}

COIN_ICONS = {
    "BTC-USD": "https://s2.coinmarketcap.com/static/img/coins/64x64/1.png",
    "ETH-USD": "https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png",
    "DOGE-USD": "https://s2.coinmarketcap.com/static/img/coins/64x64/74.png",
    "SHIB-USD": "https://s2.coinmarketcap.com/static/img/coins/64x64/5994.png",
    "FLOKI-USD": "https://s2.coinmarketcap.com/static/img/coins/64x64/10804.png"
}

def format_big_number(num):
    """Format angka besar"""
    if num is None or num == 0:
        return "-"
    if num >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return f"${num:,.0f}"
    
def format_price(num):
    """Format harga"""
    if num is None or num == 0:
        return "$0.00"
    if num < 0.0000001:
        return f"${num:.12f}"
    elif num < 1:
        return f"${num:.8f}"
    else:
        return f"${num:,.2f}"

@st.cache_data(ttl=600)
def get_market_summary():
    """Mengambil data dan waktu pengambilan data"""
    fetch_time = datetime.now().strftime("%H:%M:%S")
    summary_data = []

    for ticker, name in COINS.items():
        try:
            t = yf.Ticker(ticker)
            
            # mengambil info dengan error handling
            try:
                info = t.info
            except:
                info = {}

            # mengambil history
            hist = t.history(period="max", interval="1d")
            
            if hist.empty:
                continue

            # logika Fallback Harga (Jika info kosong, ambil dari history)
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or hist['Close'].iloc[-2]
            
            # Hitung Change %
            if prev_close and prev_close > 0:
                change_pct = ((current_price - prev_close) / prev_close) * 100
            else:
                change_pct = 0.0

            # Ambil Market Cap & Volume
            market_cap = info.get('marketCap', 0)
            volume = info.get('volume24Hr', 0) or info.get('volume', 0)
            if not hist.empty:
                valid_lows = hist.loc[hist['Low'] > 0, 'Low']
                if not valid_lows.empty:
                    atl = valid_lows.min()
                else:
                    atl = 0
            else:
                atl = 0

            summary_data.append({
                "Ticker": ticker,
                "Name": name,
                "Icon": COIN_ICONS.get(ticker, ""),
                "Price": current_price,
                "Change": change_pct,
                "ATL": atl,
                "MarketCap": market_cap,
                "Volume": volume,
            })

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
    
    return summary_data, fetch_time

@st.cache_data(ttl=3600)
def get_data_with_indikacators(ticker, start, end, interval):
    """Mengambil data historis dan melakukan feature engineering dengan aman"""
    try:
        # PERBAIKAN MLOps: Gunakan Ticker().history() karena lebih stabil dari download()
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval=interval)

        if df is None or df.empty:
            return pd.DataFrame()

        # Penting untuk Plotly/Streamlit: Buang zona waktu (timezone) agar tidak error saat di-plot
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Pastikan kolom yang dibutuhkan tersedia sebelum kalkulasi
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()

        # Features engineering
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # RSI (momentum)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (trend)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR (volatilitas)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        df['ATR'] = true_range.rolling(window=14).mean()

        # Buang baris kosong akibat shifting
        df.dropna(inplace=True)
        
        return df

    except Exception as e:
        print(f"Error mengambil data {ticker}: {e}")
        return pd.DataFrame()