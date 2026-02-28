import streamlit as st
import plotly.graph_objects as go
from utils import get_data_with_indikacators, get_market_summary, format_price, format_big_number, COINS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# 1. config & state
st.set_page_config(
    page_title="Detail Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    )

# validasi session state
if "selected_coin" not in st.session_state:
    st.warning("Please select a cryptocurrency from the Home page to view details.")
    st.switch_page("Home.py")

selected_coin = st.session_state['selected_coin']
coin_name = COINS.get(selected_coin, selected_coin.replace('-USD', ''))

# 2. Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    block-container {
        padding-top: 2rem;
    }
    
    /* Styling Header */
    .detail-title {
        font-size: 32px; font-weight: bold; color: white; margin-bottom: 0px;
    }
    .detail-price {
        font-size: 24px; font-weight: bold; color: #00FCE7; /* Warna Cyan terang */
        margin-right: 10px;
    }
            
    /* Styling Timeframe Selector (Radio Button horizontal) */
    div[row-widget="stRadio"] > div {
       flex-direction: row;
       align-items: center;
       justify-content: flex-start;
       gap: 10px;
    }
    div[row-widget="stRadio"] label {
        background-color: #161B22;
        padding: 5px 15px;
        border-radius: 5px;
        border: 1px solid #30363D;
        cursor: pointer; color: #8B949E; transition: all 0.3s;
    }
    div[row-widget="stRadio"] label:hover {
        border-color: #00FCE7; color: #00FCE7;
    }
    /* Menyembunyikan lingkaran radio button default */
    div[row-widget="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    
    /* Footer */
    .footer-disclaimer {
        display: flex; justify-content: space-between; align-items: flex-start;
        font-size: 12px; color: #555; margin-top: 30px;
        border-top: 1px solid #30363D; padding-top: 20px;
    }
    .footer-left { text-align: left; max-width: 48%; }
    .footer-right { text-align: right; max-width: 48%; }
</style>
""", unsafe_allow_html=True)

# 3. Logika timeframe interaktif
# helper untuk menghitung start date
def get_start_date(timeframe):
    end_date = datetime.now()
    if timeframe == '1M':
        start_date = end_date - relativedelta(months=1)
    elif timeframe == '6M':
        start_date = end_date - relativedelta(months=6)
    elif timeframe == '1Y':
        start_date = end_date - relativedelta(years=1)
    elif timeframe == 'ALL':
        start_date = datetime(2022, 1, 1)  # Tanggal awal data yang digunakan untuk training model
    else:
        start_date = end_date - timedelta(months=6)  # Default ke 6 bulan
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

# 4. Header Section

# button kembali
if st.button("← Back to Market Overview"):
    st.switch_page("Home.py")

# ambil data harga terbaru (realtime dari cache pendek)
market_data, _ = get_market_summary()
coin_info = next((item for item in market_data if item ['Ticker'] == selected_coin), None)
current_price = coin_info['Price'] if coin_info else 0

col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.markdown(f"<div class='detail-title'>{coin_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='detail-price'>{format_price(current_price)}</div>", unsafe_allow_html=True)

with col_header_2:
    st.markdown(f"<div style='height: 15px'></div>", unsafe_allow_html=True)  # Spacer
    if st.button(f"Start Prediction", type="primary", use_container_width=True):
        st.switch_page("pages/Prediction.py")

st.divider()

# 5. Chart Interaktif
st.write("### Price Chart")
timeframe_selected = st.radio(
    "Select Timeframe:",
    options=['1M', '6M', '1Y', 'ALL'],
    index=1,  # Default ke 6 bulan
    horizontal=True,
    label_visibility="collapsed"
)

start_date, end_date = get_start_date(timeframe_selected)

with st.spinner(f"Loading data {timeframe_selected}..."):
    df = get_data_with_indikacators(selected_coin, start_date, end_date, interval="1d")

if not df.empty:
    fig = go.Figure()

    # candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Harga',
        increasing_line_color='#00FF00',
        decreasing_line_color='#FF0000',
    ))

    # update Layout agar bersih dan profesional
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',

        xaxis=dict(
            showgrid=False,
            color='#8B949E',
            rangeslider=dict(visible=False),
            type='date',
        ),
        yaxis=dict(
            side='right',
            showgrid=True,
            gridcolor='#161B22',
            zeroline=False,
            color='#8B949E',
            tickprefix="$",
        ),
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    st.caption(f"Menampilkan data historis dari {start_date} hingga {end_date}. Gunakan mouse untuk zoom dan pan pada grafik.")

else:
    st.warning(f"No data available for the selected timeframe {timeframe_selected}.")