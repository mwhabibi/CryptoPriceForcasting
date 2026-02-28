import os
import time
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
import json

metrics_dict = {}

# Matikan peringatan agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# SETUP PATH (LOKASI FILE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALERS_DIR = os.path.join(BASE_DIR, 'scalers')
OUTPUT_DIR = os.path.join(BASE_DIR, 'hasil_ujicobamodel')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Buka file dengan mode 'w' (write)
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message) # Tulis ke Layar
        self.log.write(message)      # Tulis ke File

    def flush(self):
        # Dibutuhkan untuk kompatibilitas sistem python
        self.terminal.flush()
        self.log.flush()

# Aktifkan Pencatatan Otomatis
log_filename = os.path.join(OUTPUT_DIR, "log_ujicobamodel.txt")
sys.stdout = DualLogger(log_filename)

print(f"Working Directory Script: {BASE_DIR}")
print(f"Folder Models terdeteksi di: {MODELS_DIR}")
print(f"Folder Scalers terdeteksi di: {SCALERS_DIR}")
print(f"Gambar akan disimpan di: {OUTPUT_DIR}")

# KONFIGURASI
COINS = ["BTC-USD", "ETH-USD", "DOGE-USD", "SHIB-USD", "FLOKI-USD"]
START_BUFFER = "2025-10-01"
TEST_START   = "2026-01-01"
TEST_END     = "2026-01-21"
DOWNLOAD_END = "2026-01-25"
LOOKBACK     = 60

# 1. FUNGSI AMBIL DATA & HITUNG INDIKATOR
def get_data_with_indicators(ticker, start, end):
    print(f"Downloading data {ticker}...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        raise ValueError(f"Data kosong untuk {ticker}")

    # FEATURE ENGINEERING
    # Log Return
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    df.dropna(inplace=True)
    return df

# EKSEKUSI PENGUJIAN UTAMA
print("\n" + "="*70)
print(f"MEMULAI PENGUJIAN VALIDASI MODEL (1-21 Jan 2026)")
print("="*70)

for ticker in COINS:
    print(f"\nAnalisis Koin: {ticker}")
    print("Jeda 3 detik agar aman dari blokir Yahoo Finance API...")
    time.sleep(3) 

    try:
        # A. LOAD FILE PENTING
        model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.keras")
        scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error Path: File tidak ditemukan di {model_path}")
            continue
            
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # B. AMBIL DATA LENGKAP
        df_full = get_data_with_indicators(ticker, START_BUFFER, DOWNLOAD_END)
        
        # C. TENTUKAN TANGGAL UJI COBA MODEL PREDIKSI
        mask = (df_full.index >= TEST_START) & (df_full.index <= TEST_END)
        test_dates = df_full.loc[mask].index
        
        if len(test_dates) == 0:
            print("Data kosong pada range tanggal tersebut.")
            continue

        actual_prices = []
        predicted_prices = []
        dates_plot = []
        
        print(f"Melakukan simulasi prediksi...")
        
        # D. LOOP PREDIKSI HARIAN
        for date in test_dates:
            idx = df_full.index.get_loc(date)
            
            if idx < LOOKBACK:
                continue
            
            input_window = df_full.iloc[idx-LOOKBACK : idx]
            features = ['Log_Ret', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volume']
            input_values = input_window[features].values
            
            input_scaled = scaler.transform(input_values)
            input_reshaped = np.expand_dims(input_scaled, axis=0)
            
            pred_log_ret_scaled = model.predict(input_reshaped, verbose=0)[0][0]
            
            scale_factor = scaler.scale_[0] 
            min_factor = scaler.min_[0]    
            pred_log_ret = (pred_log_ret_scaled - min_factor) / scale_factor
            
            last_close_price = input_window['Close'].iloc[-1]
            pred_price = last_close_price * np.exp(pred_log_ret)
            
            actual_price = df_full.loc[date, 'Close']
            
            dates_plot.append(date)
            actual_prices.append(actual_price)
            predicted_prices.append(pred_price)

        # E. HITUNG ERROR
        if len(actual_prices) > 0:
            rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
            mae = mean_absolute_error(actual_prices, predicted_prices)
            mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
            accuracy = 100 * (1 - mape)
            
            print(f"HASIL AKHIR (1-21 Jan 2026):")
            print(f"RMSE    : ${rmse:.4f}")
            print(f"MAE     : ${mae:.4f}")
            print(f"MAPE    : {mape:.2%}")
            print(f"AKURASI : {accuracy:.2f}%")

            metrics_dict[ticker] = {
                "RMSE": round(float(rmse), 8),
                "MAPE": round(float(mape * 100), 2)
            }

            plt.figure(figsize=(12, 6))
            plt.plot(dates_plot, actual_prices, label='Actual (Real)', color='green', marker='o')
            plt.plot(dates_plot, predicted_prices, label='Predicted (AI)', color='red', linestyle='--', marker='x')
            plt.title(f"{ticker} - Validasi Model ({TEST_START} s.d {TEST_END})")
            plt.xlabel("Tanggal")
            plt.ylabel("Harga (USD)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            # plt.show()

            filename = f"{ticker}_ujicobamodel.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(filepath)
            print(f"Gambar grafik disimpan di: {filepath}")
            plt.close()
            
        else:
            print("Tidak ada data prediksi yang dihasilkan.")
        
        print("-" * 70)

    except Exception as e:
        print(f"CRITICAL ERROR pada {ticker}: {e}")
        import traceback
        traceback.print_exc()

metrics_path = os.path.join(BASE_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)
print(f"\n✅ File metrik berhasil disimpan di: {metrics_path}")

print("\nPENGUJIAN SELESAI.")