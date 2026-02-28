import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from utils import COINS, format_price

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="Prediction Result", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

if 'selected_coin' not in st.session_state:
    st.session_state['selected_coin'] = 'BTC-USD'

selected_coin = st.session_state['selected_coin']

# --- 2. CUSTOM CSS (Sesuai Mockup) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .header-title { font-size: 28px; font-weight: bold; }
    
    /* Styling Metrics */
    .metric-container { margin-bottom: 20px; }
    .metric-title { font-size: 14px; color: #8B949E; font-weight: bold; }
    .metric-value { font-size: 16px; color: #E3B341; font-weight: bold; } /* Warna oranye/emas */
    .metric-sub { font-size: 12px; color: #8B949E; font-style: italic; }
    
    /* Tabel Kustom */
    .pred-table { width: 100%; border-collapse: collapse; margin-top: 15px; color: white; font-size: 14px; }
    .pred-table th { text-align: right; padding: 12px; border-bottom: 1px solid #30363D; color: #8B949E; }
    .pred-table th:first-child { text-align: left; }
    .pred-table td { text-align: right; padding: 12px; border-bottom: 1px solid #21262D; }
    .pred-table td:first-child { text-align: left; font-weight: bold; }
    .change-up { color: #00FF00; }
    .change-down { color: #FF4B4B; }
    
    /* Footer */
    .footer { display: flex; justify-content: space-between; font-size: 12px; color: #8B949E; margin-top: 40px; border-top: 1px solid #30363D; padding-top: 20px; }
    .footer-left { max-width: 45%; }
    .footer-right { max-width: 45%; text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD METRICS & MODEL ---
@st.cache_data
def load_metrics():
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

metrics_data = load_metrics()
coin_metrics = metrics_data.get(selected_coin, {"RMSE": 0, "MAPE": 0})

@st.cache_resource
def load_ml_assets(ticker):
    try:
        model = load_model(f"models/{ticker}_best_model.keras")
        scaler = joblib.load(f"scalers/{ticker}_scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

# --- 4. TOP HEADER LAYOUT ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='header-title'>Prediction Result</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='display:flex; justify-content:flex-end; gap:10px;'>", unsafe_allow_html=True)
    if st.button("❮ Back To Previous"):
        st.switch_page("pages/Detail.py")
    if st.button("Re-Analysis (Re-Run)", type="primary"):
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Metrics Display
st.markdown(f"""
<div class="metric-container">
    <div>
        <span class="metric-title">RMSE Score:</span> <span class="metric-value">{coin_metrics['RMSE']:,}</span><br>
        <span class="metric-sub">Lower is better</span>
    </div>
    <div style="margin-top: 10px;">
        <span class="metric-title">MAPE Score:</span> <span class="metric-value">{coin_metrics['MAPE']}%</span><br>
        <span class="metric-sub">Average error percentage</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. LOGIKA PREDIKSI (Bypass yfinance bug dengan history period="max") ---
model, scaler = load_ml_assets(selected_coin)

if model is None or scaler is None:
    st.error("Model atau Scaler tidak ditemukan. Pastikan file ada di folder 'models' dan 'scalers'.")
    st.stop()

with st.spinner("Memproses algoritma LSTM..."):
    # Trik Bypass yfinance: Tarik data maksimum, lalu potong
    t = yf.Ticker(selected_coin)
    df = t.history(period="1y", interval="1d") # Tarik 1 tahun terakhir agar pasti cukup
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Feature Engineering Cepat
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    
    df.dropna(inplace=True)

    # Persiapan Input Model
    LOOKBACK = 60
    FEATURES = ['Log_Ret', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volume']
    
    recent_data = df[FEATURES].values[-LOOKBACK:]
    scaled_data = scaler.transform(recent_data)
    X_input = scaled_data.reshape(1, LOOKBACK, len(FEATURES))
    
    # Inferensi
    pred_scaled = model.predict(X_input, verbose=0)[0]
    dummy_array = np.zeros((7, len(FEATURES)))
    dummy_array[:, 0] = pred_scaled
    pred_log_ret = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Konversi Harga
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    
    future_dates = []
    future_prices = []
    changes_pct = []
    
    current_p = last_price
    for i, log_r in enumerate(pred_log_ret):
        next_p = current_p * np.exp(log_r)
        # Hitung persentase perubahan dari hari sebelumnya
        change = ((next_p - current_p) / current_p) * 100 
        
        future_prices.append(next_p)
        changes_pct.append(change)
        future_dates.append(last_date + timedelta(days=i+1))
        
        current_p = next_p

# --- 6. VISUALISASI CHART (Future Projection) ---
# Menggabungkan Data Aktual Terakhir & Prediksi untuk Grafik yang Mulus
plot_dates = list(df.index[-60:]) + future_dates
plot_prices = list(df['Close'].iloc[-60:]) + future_prices

fig = go.Figure()
# Garis Harga Asli (Biru Tua)
fig.add_trace(go.Scatter(
    x=df.index[-60:], y=df['Close'].iloc[-60:],
    mode='lines', name='Actual Price', line=dict(color='#4A72B2', width=2)
))
# Garis Prediksi (Merah Putus-putus)
fig.add_trace(go.Scatter(
    x=[df.index[-1]] + future_dates, y=[df['Close'].iloc[-1]] + future_prices,
    mode='lines', name='Predicted Price', line=dict(color='#FF4B4B', width=2, dash='dash')
))

fig.update_layout(
    title=f"{selected_coin} Actual vs. Predicted Prices (Future Projection)",
    height=400,
    plot_bgcolor='#E6E6EA', # Background abu-abu terang seperti di mockup
    paper_bgcolor='#0E1117',
    font=dict(color='black'),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridcolor='white'),
    yaxis=dict(showgrid=True, gridcolor='white', title="Price (USD)"),
    showlegend=True,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)
st.plotly_chart(fig, use_container_width=True)

# --- 7. TABEL PREDIKSI ---
st.markdown("### Predicted Prices (Next 7 Days)")

# Membuat HTML Table agar sama persis dengan mockup
table_html = '<table class="pred-table">'
table_html += '<thead><tr><th>Date</th><th>Price</th><th>Change (%)</th></tr></thead><tbody>'

for i in range(7):
    date_str = future_dates[i].strftime('%d %b %Y')
    price_str = format_price(future_prices[i])
    change_val = changes_pct[i]
    
    color_class = "change-up" if change_val >= 0 else "change-down"
    sign = "+" if change_val >= 0 else ""
    
    table_html += f"<tr><td>{date_str}</td><td>{price_str}</td><td class='{color_class}'>{sign}{change_val:.2f}%</td></tr>"

table_html += '</tbody></table>'
st.markdown(table_html, unsafe_allow_html=True)

# --- 8. FOOTER ---
st.markdown("""
<div class="footer">
    <div class="footer-left">
        <b>System Information & Disclaimer</b><br>
        This System Uses A Long Short-Term Memory (LSTM) Algorithm To Predict Crypto Asset Prices Based On Historical Data. Accuracy Is Evaluated Using RMSE And MAPE Metrics.
    </div>
    <div class="footer-right">
        <b style="color: white;">Important Notice</b><br>
        Content Provided In This Dashboard Is For Informational Purposes Only And Does Not Constitute Financial Advice, Investment Recommendation, Or Trading Endorsement. Cryptocurrency Trading Involves High Risk And Volatility. Please Conduct Your Own Research (DYOR) Before Trading.
    </div>
</div>
""", unsafe_allow_html=True)