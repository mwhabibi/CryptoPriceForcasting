import streamlit as st
import pandas as pd
from utils import get_market_summary, format_big_number, format_price

# config page
st.set_page_config(
    page_title="Crypto Market Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }

    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
        text-align: left;
        margin-bottom: 10px;
        min-height: 120px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .coin-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }

    .coin-logo {
        width: 30px;
        height: 30px;
        border-radius: 50%; 
        object-fit: cover;
    }
    .coin-name {
        color: #8B949E;
        font-size: 14px;
        font-weight: bold;
    }            
    .coin-price {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: bold;
        margin: 5px 0;
    }
    .coin-change-up { color: #00FF00; font-size: 12px; }
    .coin-change-down { color: #FF4B4B; font-size: 12px; }
    
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        color: white;
        font-family: sans-serif;
        margin-top: 5px;
    }
    .styled-table th {
        text-align: right;
        padding: 12px 15px;
        color: #8B949E;
        font-weight: normal;
        border-bottom: 1px solid #30363D;
    }
    .styled-table th:first-child { text-align: left; }
    
    .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #21262D;
        text-align: right;
    }
    .styled-table td:first-child { text-align: left; font-weight: bold; display: flex; align-items: center; gap: 8px;}
    
    .ticker-symbol { color: #8B949E; font-size: 0.8em; margin-left: 5px; }

    /* Footer Disclaimer */
    .footer-disclaimer {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        font-size: 14px;
        color: #555;
        margin-top: 50px;
        margin-bottom: 0;
        border-top: 1px solid #333;
        padding-top: 10px;
    }
            
    .footer-left{
        text-align: left;
        max-width: 48%;
    }

    .footer-right{
        text-align: right;
        max-width: 48%;   
    }
            
    
</style>
""", unsafe_allow_html=True)

# load data
with st.spinner("Loading market data..."):
    data, last_updated = get_market_summary()

# header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h2 style='margin-bottom:0; padding-bottom:0;'>CRYPTOCURRENCY PRICE PREDICTION</h2>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:0; padding-top:0;'>Market Overview</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color: #8B949E;'>Select an asset below to view historical data and run model price prediction.</p>",
                unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""<p style='text-align: right; color: #8B949E; margin-top:20px;'>Last Updated: {last_updated}</p>""",
                unsafe_allow_html=True)
    
# CARDS (ROW 1)
if not data:
    st.error("Gagal mengambil data. Cek koneksi internet.")
else:
    cols = st.columns(len(data))
    for i, item in enumerate(data):
        color_class = "coin-change-up" if item["Change"] >= 0 else "coin-change-down"
        arrow = "▲" if item["Change"] >= 0 else "▼"
        
        with cols[i]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="coin-header">
                        <img src="{item['Icon']}" class="coin-logo" onerror="this.style.display='none'">
                        <span class="coin-name">{item['Ticker'].replace('-USD','')}</span>
                    </div>
                    <div>
                        <div class="coin-price">{format_price(item['Price'])}</div>
                        <div class="{color_class}">
                            {arrow} {item['Change']:.2f}% <span style='opacity:0.6; font-size:0.9em'>(24h)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"Analyze {item['Name']}", key=f"btn_{i}", use_container_width=True):
                 st.session_state['selected_coin'] = item['Ticker']
                #  st.switch_page("pages/Detail.py")

# MARKET TABLE (ROW 2)
st.markdown("###")
table_html = '<table class="styled-table">'
table_html += '<thead><tr><th style="text-align: left;">Asset</th><th style="text-align: right;">Price</th><th style="text-align: right;">24h Change</th><th style="text-align: right;">All-Time Low</th><th style="text-align: right;">Market Cap</th><th style="text-align: right;">Volume (24h)</th></tr></thead>'
table_html += '<tbody>'

for item in data:
    change_color = "#00FF00" if item['Change'] >= 0 else "#FF4B4B"
    table_html += f"""
    <tr>
        <td style="text-align: left;">
            <img src="{item['Icon']}" style="width:20px; height:20px; border-radius:50%; vertical-align:middle; margin-right:5px;">
            {item['Name']} <span class='ticker-symbol'>{item['Ticker'].replace('-USD','')}</span>
        </td>
        <td style="text-align: right;">{format_price(item['Price'])}</td>
        <td style="text-align: right; color: {change_color};">{item['Change']:.2f}%</td>
        <td style="text-align: right;">{format_price(item['ATL'])}</td>
        <td style="text-align: right;">{format_big_number(item['MarketCap'])}</td>
        <td style="text-align: right;">{format_big_number(item['Volume'])}</td>
    </tr>"""

table_html += '</tbody></table>'
st.markdown(table_html, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer-disclaimer">
<div class="footer-left">
    <b>System Information & Disclaimer</b><br>
    This System Uses A Long Short-Term Memory (LSTM) Algorithm To Predict Crypto Asset Prices Based On Historical Data. 
    Accuracy Is Evaluated Using MAE, RMSE, And MAPE Metrics.
</div>
    
<div class="footer-right">
    <b>Important Notice</b><br>
    Content Provided In This Dashboard Is For Informational Purposes Only And Does Not Constitute Financial Advice. 
    Cryptocurrency Trading Involves High Risk And Volatility. Please Conduct Your Own Research (DYOR).
</div>
</div>
""", unsafe_allow_html=True)