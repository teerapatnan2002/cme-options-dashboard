import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# ⚙️ ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="CME Quant Dashboard", layout="wide")
st.title("🎯 CME Options Dashboard & AI Analysis")
st.markdown("ศูนย์บัญชาการวิเคราะห์โครงสร้างตลาด (OI, Max Pain, Net GEX พร้อม AI Quant Analyst)")
st.caption("📡 Dashboard อ่านข้อมูลจาก Firebase | 🤖 Bot รันจากเครื่อง Local (Mac)")

FIREBASE_URL = "https://cme-quant-bot-default-rtdb.asia-southeast1.firebasedatabase.app/daily_data.json"

# =========================================================
# ⏰ Expiration Time Utilities (Thai Time / CME CT)
# =========================================================
def is_us_dst(dt):
    year = dt.year
    mar1 = datetime(year, 3, 1)
    days_to_sun = (6 - mar1.weekday()) % 7
    dst_start = mar1 + timedelta(days=days_to_sun + 7)
    nov1 = datetime(year, 11, 1)
    days_to_sun = (6 - nov1.weekday()) % 7
    dst_end = nov1 + timedelta(days=days_to_sun)
    return dst_start <= dt.replace(hour=0, minute=0, second=0, microsecond=0) < dst_end

def get_expiration_datetime_thai(exp_date):
    if is_us_dst(exp_date):
        return exp_date + timedelta(hours=24, minutes=30)
    else:
        return exp_date + timedelta(days=1, hours=1, minutes=30)

def is_series_active(exp_date, now=None):
    if now is None:
        now = datetime.now()
    return now < get_expiration_datetime_thai(exp_date)

# 🤖 AI Configuration (Groq Only)
GROQ_API_KEY = "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"
GROQ_MODEL = "openai/gpt-oss-120b"

@st.cache_data(ttl=300)
def fetch_data():
    response = requests.get(FIREBASE_URL)
    if response.status_code == 200:
        return response.json()
    return None

def calculate_gamma(F, K, T, sigma):
    if T <= 0 or sigma <= 0 or F <= 0:
        return 0.0
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (F * sigma * np.sqrt(T))
    return gamma

def process_gex(df_call, df_put, F, DTE, sigma):
    T = DTE / 365.0
    df_call = df_call.copy()
    df_put = df_put.copy()
    df_call['Gamma'] = df_call['Strike'].apply(lambda K: calculate_gamma(F, K, T, sigma))
    df_put['Gamma'] = df_put['Strike'].apply(lambda K: calculate_gamma(F, K, T, sigma))
    df_call['GEX'] = df_call['Gamma'] * df_call['Value'] * F * 100
    df_put['GEX'] = df_put['Gamma'] * df_put['Value'] * F * 100 * -1
    gex_df = pd.merge(df_call[['Strike', 'GEX']], df_put[['Strike', 'GEX']], on='Strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)
    gex_df['Net_GEX'] = gex_df['GEX_Call'] + gex_df['GEX_Put']
    return gex_df.sort_values('Strike')

def calculate_max_pain(df_call, df_put):
    all_strikes = sorted(list(set(df_call['Strike'].tolist() + df_put['Strike'].tolist())))
    pain_data = []
    for test_strike in all_strikes:
        call_loss = df_call[df_call['Strike'] < test_strike].apply(lambda x: (test_strike - x['Strike']) * x['Value'], axis=1).sum()
        put_loss = df_put[df_put['Strike'] > test_strike].apply(lambda x: (x['Strike'] - test_strike) * x['Value'], axis=1).sum()
        pain_data.append({'Strike': test_strike, 'Call Pain': call_loss, 'Put Pain': put_loss, 'Total Pain': call_loss + put_loss})
    pain_df = pd.DataFrame(pain_data)
    max_pain_strike = pain_df.loc[pain_df['Total Pain'].idxmin()]['Strike']
    return pain_df, max_pain_strike

def get_ai_analysis(current_price, max_pain, gex_df, dte, series_name, vol_call_df=None, vol_put_df=None):
    try:
        strongest_resistance = gex_df.loc[gex_df['GEX_Call'].idxmax()]['Strike']
        strongest_support = gex_df.loc[gex_df['GEX_Put'].idxmin()]['Strike']
    except:
        strongest_resistance = "N/A"
        strongest_support = "N/A"
    vol_analysis_txt = "ไม่มีข้อมูล Intraday Volume"
    if vol_call_df is not None and not vol_call_df.empty:
        try:
            top_call_vol = vol_call_df.nlargest(3, 'Value')[['Strike', 'Value']].to_dict('records')
            top_put_vol = vol_put_df.nlargest(3, 'Value')[['Strike', 'Value']].to_dict('records')
            vol_analysis_txt = f"Top Call Vol: {top_call_vol}, Top Put Vol: {top_put_vol}"
        except:
            vol_analysis_txt = "Error extracting volume data"
    prompt = f"""Senior Quantitative Analyst... (omitted prompt content for brevity)"""
    # ⚡ Groq API Call
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"messages": [{"role": "user", "content": prompt}], "model": GROQ_MODEL, "temperature": 0.7}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200: return response.json()['choices'][0]['message']['content']
        else: return f"❌ Groq API Error: {response.text}"
    except Exception as e: return f"❌ Connection Error: {e}"

# ==========================================
# 📊 หน้าจอ Dashboard (Read-Only from Firebase)
# ==========================================
data = fetch_data()
if data:
    st.sidebar.success("✅ เชื่อมต่อ Firebase สำเร็จ!")
    st.sidebar.caption("💡 ข้อมูลอัพเดทจาก Bot ที่รันบนเครื่อง Local")
    available_dates = list(data.keys())
    selected_date = st.sidebar.selectbox("📅 เลือกวันที่", sorted(available_dates, reverse=True))
    available_series = list(data[selected_date].keys())
    series_display_map = {}; series_expiration_info = {}; now = datetime.now()
    for s in available_series:
        try:
            first_time_key = list(data[selected_date][s].keys())[0]
            dte = data[selected_date][s][first_time_key].get('dte', 0)
            # ... rest of the series processing ...
        except: pass
    # ... rest of the app logic ...
else: st.error("❌ ไม่สามารถดึงข้อมูลจาก Firebase ได้")
