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
# เซนเซอร์ API Key ไว้เพื่อความปลอดภัย (GitHub Secret Scanning)
# ต้องตั้งค่า Secrets ใน Streamlit Cloud -> App Settings -> Secrets แทน
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "NO_API_KEY_FOUND_IN_SECRETS"
GROQ_MODEL = "openai/gpt-oss-120b"

@st.cache_data(ttl=300) 
def fetch_data():
    response = requests.get(FIREBASE_URL)
    if response.status_code == 200:
        return response.json()
    return None

# ==========================================
# 🧠 ฟังก์ชันคณิตศาสตร์: Black-Scholes & GEX
# ==========================================
def calculate_greeks(S, K, T, v, r, opt_type):
    if T <= 0 or v <= 0 or S <= 0: return 0.0, 0.0, 0.0
    d1 = (np.log(S/K) + (r + 0.5 * v**2)*T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    
    if opt_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * v * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * v) / (2 * np.sqrt(T)))
    return delta, gamma, theta

def process_gex(df_call, df_put, F, DTE, sigma, vol_call_df=None, vol_put_df=None):
    T = DTE / 365.0 
    df_call = df_call.copy()
    df_put = df_put.copy()
    
    # Calculate Greeks (Delta, Gamma, Theta) for Call options
    call_greeks = df_call['Strike'].apply(lambda K: calculate_greeks(F, K, T, sigma, 0.0, 'call'))
    df_call['Delta'] = call_greeks.apply(lambda x: x[0])
    df_call['Gamma'] = call_greeks.apply(lambda x: x[1])
    df_call['Theta'] = call_greeks.apply(lambda x: x[2])
    
    # Calculate Greeks for Put options
    put_greeks = df_put['Strike'].apply(lambda K: calculate_greeks(F, K, T, sigma, 0.0, 'put'))
    df_put['Delta'] = put_greeks.apply(lambda x: x[0])
    df_put['Gamma'] = put_greeks.apply(lambda x: x[1])
    df_put['Theta'] = put_greeks.apply(lambda x: x[2])
    
    # Standard GEX (Based on Open Interest value)
    df_call['GEX'] = df_call['Gamma'] * df_call['Value'] * F * 100
    df_put['GEX'] = df_put['Gamma'] * df_put['Value'] * F * 100 * -1 
    
    # Merge and calculate Net GEX
    gex_df = pd.merge(df_call[['Strike', 'GEX', 'Delta', 'Gamma', 'Theta']], 
                      df_put[['Strike', 'GEX', 'Delta', 'Gamma', 'Theta']], 
                      on='Strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)
    gex_df['Net_GEX'] = gex_df['GEX_Call'] + gex_df['GEX_Put']
    
    # Volume-Weighted GEX (V-GEX)
    gex_df['VGEX_Call'] = 0.0
    gex_df['VGEX_Put'] = 0.0
    gex_df['Net_VGEX'] = 0.0
    
    if vol_call_df is not None and not vol_call_df.empty and vol_put_df is not None and not vol_put_df.empty:
        # Merge Intraday Volume data
        vol_c = vol_call_df[['Strike', 'Value']].rename(columns={'Value': 'Vol_Call'})
        vol_p = vol_put_df[['Strike', 'Value']].rename(columns={'Value': 'Vol_Put'})
        gex_df = pd.merge(gex_df, vol_c, on='Strike', how='left').fillna(0)
        gex_df = pd.merge(gex_df, vol_p, on='Strike', how='left').fillna(0)
        
        # Calculate V-GEX = Gamma * Volume
        gex_df['VGEX_Call'] = gex_df['Gamma_Call'] * gex_df['Vol_Call'] * F * 100
        gex_df['VGEX_Put'] = gex_df['Gamma_Put'] * gex_df['Vol_Put'] * F * 100 * -1
        gex_df['Net_VGEX'] = gex_df['VGEX_Call'] + gex_df['VGEX_Put']
        
    return gex_df.sort_values('Strike')

def find_optimal_credit_spreads(current_price, gamma_wall, support_floor, gex_df, min_strike_width=50):
    if gex_df.empty or current_price <= 0: return None, None
    bear_call_spread = None
    bull_put_spread = None
    
    # Bear Call Spread (Above Gamma Wall)
    call_candidates = gex_df[gex_df['Strike'] >= gamma_wall].copy()
    if not call_candidates.empty and len(call_candidates) >= 2:
        # Sort by Theta (most negative = highest decay rate for shorting)
        call_candidates = call_candidates.sort_values('Theta_Call', ascending=True)
        short_strike = call_candidates.iloc[0]['Strike']
        
        # Find Long Strike for protection
        long_candidates = call_candidates[call_candidates['Strike'] >= short_strike + min_strike_width]
        if not long_candidates.empty:
            long_strike = long_candidates.iloc[0]['Strike']
            bear_call_spread = {'type': 'Bear Call', 'short': short_strike, 'long': long_strike, 'theta_edge': call_candidates.iloc[0]['Theta_Call']}
            
    # Bull Put Spread (Below Support Floor)
    put_candidates = gex_df[(gex_df['Strike'] <= support_floor) & (gex_df['Strike'] > 0)].copy()
    if not put_candidates.empty and len(put_candidates) >= 2:
        put_candidates = put_candidates.sort_values('Theta_Put', ascending=True)
        short_strike = support_floor # Use the actual floor as the short strike
        long_candidates = put_candidates[put_candidates['Strike'] <= short_strike - min_strike_width]
        if not long_candidates.empty:
            long_strike = long_candidates.iloc[0]['Strike']
            bull_put_spread = {'type': 'Bull Put', 'short': short_strike, 'long': long_strike, 'theta_edge': put_candidates[put_candidates['Strike'] == short_strike]['Theta_Put'].values[0] if short_strike in put_candidates['Strike'].values else 'Optimized'}
            
    return bear_call_spread, bull_put_spread

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

def calculate_iv_skew(df_call, df_put, current_price):
    if df_call.empty or df_put.empty or current_price <= 0: return None, None, None
    
    # หาสัญญาที่ OTM ประมาณ 5%
    otm_call_strike = current_price * 1.05
    otm_put_strike = current_price * 0.95
    
    # ดึงค่า IV ที่ใกล้เคียง Strike ที่กำหนดที่สุด
    try:
        call_iv = df_call.iloc[(df_call['Strike'] - otm_call_strike).abs().argsort()[:1]]['vol_settle'].values[0]
        put_iv = df_put.iloc[(df_put['Strike'] - otm_put_strike).abs().argsort()[:1]]['vol_settle'].values[0]
        skew_ratio = put_iv / call_iv if call_iv > 0 else 1.0
        return call_iv, put_iv, skew_ratio
    except:
        return None, None, None

def calculate_spi(current_price, gamma_wall, max_pain, gex_df):
    if not isinstance(gamma_wall, (int, float)) or gamma_wall <= 0: return 0.0
    if gex_df.empty: return 0.0
    
    # 1. ระยะห่าง Spot ถึง Gamma Wall (ยิ่งใกล้ยิ่งเสี่ยง)
    distance_pct = abs(current_price - gamma_wall) / current_price
    
    # ให้คะแนนความใกล้ (ใกล้มาก = 100, ห่าง > 2% = 0)
    proximity_score = max(0, 100 - (distance_pct * 100 / 0.02))
    
    # 2. ความสูงของกราฟ Gamma Wall เทียบกับค่าเฉลี่ย
    avg_call_gex = gex_df['GEX_Call'].mean()
    wall_gex = gex_df.loc[gex_df['Strike'] == gamma_wall, 'GEX_Call'].values
    wall_gex_val = wall_gex[0] if len(wall_gex) > 0 else avg_call_gex
    
    # ให้คะแนนความหนาแน่น
    density_ratio = wall_gex_val / avg_call_gex if avg_call_gex > 0 else 1.0
    density_score = min(100, density_ratio * 10) # normalization แบบง่ายๆ
    
    # ถ้าราคาอยู่เหนือ Gamma Wall แล้ว ให้ SPI พุ่งสูงเป็นพิเศษ (Breakout Trap)
    breakout_multiplier = 1.2 if current_price > gamma_wall else 1.0
    
    # รวมคะแนนเป็น % สัมพัทธ์
    spi_raw = ((proximity_score * 0.6) + (density_score * 0.4)) * breakout_multiplier
    return min(100.0, round(spi_raw, 2))

# ==========================================
# 🤖 ฟังก์ชัน AI Quant Analyst (Custom Prompt)
# ==========================================
def get_ai_analysis(current_price, max_pain, gex_df, dte, series_name, vol_call_df=None, vol_put_df=None, spi_score=0.0, skew_ratio=1.0, bear_call=None, bull_put=None, simulated_shift=0.0, simulated_price=0.0):
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
            
    skew_analysis = f"Puts are {((skew_ratio - 1) * 100):.1f}% more expensive than Calls (Overvalued)" if skew_ratio > 1 else f"Calls are {((1 - skew_ratio) * 100):.1f}% more expensive than Puts (Overvalued)"
    
    bear_strat = f"Short {bear_call['short']} / Long {bear_call['long']} (Edge: {isinstance(bear_call['theta_edge'], str) and bear_call['theta_edge'] or format(bear_call['theta_edge'], '.3f')})" if bear_call else "N/A"
    bull_strat = f"Short {bull_put['short']} / Long {bull_put['long']} (Edge: {isinstance(bull_put['theta_edge'], str) and bull_put['theta_edge'] or format(bull_put['theta_edge'], '.3f')})" if bull_put else "N/A"

    prompt = f"""
    บทบาทและตรรกะหลัก (Role & Core Logic):
    * ทำหน้าที่เป็น 'Senior Quantitative Market Maker & Volatility Strategist' ยึดปรัชญา 'Risk is measured, not guessed.'
    * วิเคราะห์ตลาดผ่าน Liquidity, Gamma Exposure (GEX), Delta/Theta และ Volatility Surface ผ่าน Pipeline: Order flow -> Gamma Exposure -> Volatility Skew -> Tactical Execution -> Pseudo-code.
    
    =========================================
    📊 ข้อมูลอินพุตสำหรับวิเคราะห์รอบนี้ (Live Market Data):
    - Asset & Context: สัญญา {series_name}, DTE = {dte} วัน
    - Price Action: ราคาปัจจุบัน (Spot) = {current_price}
    - Intraday Volume Flow: {vol_analysis_txt}
    - Option Chain Data Summary: 
      * จุด Max Pain = {max_pain}
      * Gamma Wall (แนวต้าน Call GEX สูงสุด) = {strongest_resistance}
      * Support Floor (แนวรับ Put GEX ต่ำสุด) = {strongest_support}
    - 🧮 Advanced Quantitative Metrics:
      * Squeeze Probability Index (SPI): {spi_score}% (0=Safe, 100=Imminent Squeeze Risk)
      * IV Skew Ratio (Fear Gauge): {skew_ratio:.2f} ({skew_analysis})
    - 🎯 Tactical Execution Setup:
      * Bear Call Spread: {bear_strat}
      * Bull Put Spread: {bull_strat}
    - 🎮 Dealer Hedging Simulator Context:
      * Simulated Spot Shift: {simulated_shift:+.1f}% (Simulated Price = {simulated_price:.2f})
    =========================================
    
    📌 Theoretical Risk Assessment & Market Dynamics (For Educational Analysis Only):
    1) Flow & Squeeze Dynamics: Integrate the SPI Score with the Intraday Volume Flow. Is the smart money (Volume) attacking the Gamma Wall, or leaning on the Support Floor?
    2) Volatility Term Structure & Skew Analysis: Interpret the IV Skew Ratio. Are institutions loading up on Puts (Fear) or Calls (Greed)?
    3) Tactical Spread Setup: Review the algorithmically generated "Theta-Optimized Credit Spreads". Critique these strikes based on the Gamma boundaries and time decay.
    4) Hedging Simulator Impact: Interpret the {simulated_shift:+.1f}% spot shift. Based on the new simulated Gamma Exposure boundaries, how severe is the hedging obligation for dealers? Would they be forced to buy (short covering) or sell futures?
    5) Liquidity Analysis Theory: Explain the importance of Level 2 Order Book Depth mapping near Gamma Walls.
       
    IMPORTANT INSTRUCTION FOR OUTPUT FORMAT:
    1. First, provide your complete advanced quantitative and structural analysis in English. Use professional institutional vocabulary.
    2. Second, provide a comprehensive translation AND practical interpretation of your analysis in THAI. The Thai section should break down the complex concepts so the user can easily understand the market dynamics and educational takeaways.
    Use clear Markdown formatting with distinct headers for the English and Thai sections. Do not provide direct financial advice or trading signals.
    """
    
    # ⚡ Groq API Call
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": GROQ_MODEL,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"❌ Groq API Error: {response.text}"
    except Exception as e:
        return f"❌ Connection Error: {e}"

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
    
    # 🛠️ สร้าง Map สำหรับแสดงชื่อ Series พร้อมวันหมดอายุ + สถานะ Active/Expired
    series_display_map = {}
    series_expiration_info = {}
    now = datetime.now()
    
    for s in available_series:
        try:
            first_time_key = list(data[selected_date][s].keys())[0]
            entry = data[selected_date][s][first_time_key]
            dte = entry.get('dte', 0)
            
            # 🔑 ใช้ exp_date_str จาก CME (ถ้ามี) แทนการคำนวณจาก DTE
            exp_date_str_from_firebase = entry.get('exp_date_str', None)
            if exp_date_str_from_firebase:
                exp_date = datetime.strptime(exp_date_str_from_firebase, "%d %b %Y")
                exp_str = exp_date.strftime("%d %b %Y")
            else:
                # Fallback สำหรับข้อมูลเก่าที่ยังไม่มี exp_date_str
                selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
                exp_date = selected_dt + timedelta(days=round(dte))
                exp_str = exp_date.strftime("%d %b %Y")
            
            active = is_series_active(exp_date, now)
            cutoff = get_expiration_datetime_thai(exp_date)
            status_icon = "🟢" if active else "🔴"
            
            series_display_map[s] = f"{status_icon} {s} (Exp: {exp_str})"
            series_expiration_info[s] = {
                'exp_date': exp_date,
                'cutoff': cutoff,
                'active': active,
                'dte': dte
            }
        except:
            series_display_map[s] = s

    selected_series = st.sidebar.selectbox(
        "📌 เลือก Series", 
        available_series, 
        format_func=lambda x: series_display_map.get(x, x)
    )
    
    # ⏰ แสดงสถานะ Expiration ของ Series ที่เลือก
    if selected_series in series_expiration_info:
        info = series_expiration_info[selected_series]
        dst_label = "CDT" if is_us_dst(now) else "CST"
        cutoff_str = info['cutoff'].strftime('%d %b %Y %H:%M')
        
        if info['active']:
            remaining = info['cutoff'] - now
            hours_left = remaining.total_seconds() / 3600
            if hours_left < 24:
                st.sidebar.warning(f"⏳ Series นี้จะหมดอายุใน **{hours_left:.1f} ชม.**\n\n🕐 Cutoff: {cutoff_str} ({dst_label})")
            else:
                st.sidebar.success(f"🟢 **Series Active**\n\n🕐 Cutoff: {cutoff_str} ({dst_label})")
        else:
            st.sidebar.error(f"🔴 **Series หมดอายุแล้ว**\n\n🕐 Cutoff: {cutoff_str} ({dst_label})")
    
    available_times = list(data[selected_date][selected_series].keys())
    oi_times = [t for t in available_times if t.endswith("_OI")]
    
    if oi_times:
        selected_time = st.sidebar.selectbox("⏰ เลือกรอบเวลา", sorted(oi_times, reverse=True))
        raw_data = data[selected_date][selected_series][selected_time]
        
        # 🟢 พยายามหาข้อมูล Intraday ที่คู่กัน (หรือตัวล่าสุด)
        intraday_times = [t for t in available_times if t.endswith("_Intraday")]
        intraday_raw = None
        if intraday_times:
            latest_intraday = sorted(intraday_times)[-1]
            intraday_raw = data[selected_date][selected_series][latest_intraday]

        current_price = raw_data.get('price', 0)
        dte = raw_data.get('dte', 0)
        iv = raw_data.get('vol_settle', 0)
        
        # DataFrame ของ OI
        raw_points = raw_data["points"]
        df = pd.DataFrame(raw_points)
        df_call = df[df["SeriesName"].str.contains("Call", case=False, na=False)]
        df_put = df[df["SeriesName"].str.contains("Put", case=False, na=False)]
        
        # DataFrame ของ Intraday Volume
        df_vol_call = pd.DataFrame()
        df_vol_put = pd.DataFrame()
        
        if intraday_raw and intraday_raw.get("points"):
            df_intraday = pd.DataFrame(intraday_raw["points"])
            df_vol_call = df_intraday[(df_intraday["DataType"] == "Intraday") & (df_intraday["SeriesName"].str.contains("Call", case=False))]
            df_vol_put = df_intraday[(df_intraday["DataType"] == "Intraday") & (df_intraday["SeriesName"].str.contains("Put", case=False))]

        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ สภาวะตลาด (Parameters)**")
        st.sidebar.info(f"💵 ราคาปัจจุบัน: **{current_price}**\n\n⏳ วันหมดอายุ: **{dte} วัน**\n\n📈 ความผันผวน: **{iv:.4f}**")
        
        # ==========================================
        # 🎮 Institutional Phase 3: Dealer Hedging Simulator
        # ==========================================
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎮 Dealer Hedging Simulator")
        simulated_shift = st.sidebar.slider("Spot Price Shift (%)", -10.0, 10.0, 0.0, 0.5, help="จำลองการลากราคาขึ้นลง แล้วดูผลกระทบต่อ Gamma Wall และพฤติกรรมการ Hedge ของ Dealer")
        simulated_price = current_price * (1 + simulated_shift / 100.0)
        
        if simulated_shift != 0:
            st.sidebar.warning(f"🚀 **Simulated Spot:** {simulated_price:.2f} ({simulated_shift:+.1f}%)")

        if not df_call.empty and not df_put.empty:
            
            # --- ข้อมูลดิบพื้นฐาน ---
            pain_df, max_pain_strike = calculate_max_pain(df_call, df_put)
            
            # 1️⃣ Net GEX 
            st.subheader("🌋 Net Gamma Exposure (GEX) Profile")
            if current_price > 0 and dte > 0 and iv > 0:
                # 🛠️ ส่ง df_vol_call, df_vol_put เข้าไปคำนวณ V-GEX ด้วย (ใช้ราคา Simulated)
                gex_df = process_gex(df_call, df_put, simulated_price, dte, iv, df_vol_call, df_vol_put)
                
                # --- Quantitative Metrics (SPI & Fear Gauge) ---
                try:
                    gamma_wall = gex_df.loc[gex_df['GEX_Call'].idxmax()]['Strike']
                    support_floor = gex_df.loc[gex_df['GEX_Put'].idxmin()]['Strike']
                    spi_score = calculate_spi(simulated_price, gamma_wall, max_pain_strike, gex_df)
                    call_iv, put_iv, skew_ratio = calculate_iv_skew(df_call, df_put, simulated_price)
                except:
                    gamma_wall = simulated_price
                    support_floor = simulated_price
                    spi_score = 0.0
                    skew_ratio = 1.0
                
                # 🧮 แสดง Metrics
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    spi_color = "🔴" if spi_score > 70 else ("🟡" if spi_score > 40 else "🟢")
                    st.metric(label=f"{spi_color} Squeeze Probability Index (SPI)", value=f"{spi_score}%", 
                              help="0=ปลอดภัย, 100=เสี่ยงเกิด Gamma Squeeze สูงมาก (Spot ทะลุ Gamma Wall)")
                
                with mcol2:
                    if skew_ratio is None:
                        st.metric(label="⚖️ IV Skew Ratio", value="N/A", help="ไม่พบข้อมูล OTM Options สำหรับคำนวณ")
                        skew_ratio_for_ai = 1.0 # fallback for AI
                    else:
                        if skew_ratio > 1:
                            skew_desc = f"Puts Expensive (+{((skew_ratio-1)*100):.1f}%)"
                            skew_color = "🟥 Fear (Bearish)"
                        else:
                            skew_desc = f"Calls Expensive (+{((1-skew_ratio)*100):.1f}%)"
                            skew_color = "🟩 Greed (Bullish)"
                            
                        st.metric(label=f"⚖️ IV Skew Ratio ({skew_color})", value=f"{skew_ratio:.2f}x", delta=skew_desc, delta_color="off",
                                  help="คำนวณจาก OTM Put IV / OTM Call IV. ค่า > 1 แปลว่านักลงทุนแห่ซื้อ Puts (กลัวตลาดลง)")
                        skew_ratio_for_ai = skew_ratio
                              
                st.markdown("---")
                
                # วาดกราฟ GEX
                fig_gex = go.Figure()
                fig_gex.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX_Call'], name='Call GEX (OI)', marker_color='rgba(46, 204, 113, 0.4)'))
                fig_gex.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX_Put'], name='Put GEX (OI)', marker_color='rgba(231, 76, 60, 0.4)'))
                fig_gex.add_trace(go.Scatter(x=gex_df['Strike'], y=gex_df['Net_GEX'], name='Net GEX (OI)', mode='lines+markers', line=dict(color='yellow', width=2)))
                
                # 📊 Overlay V-GEX (Volume-Weighted)
                if 'VGEX_Call' in gex_df.columns and gex_df['VGEX_Call'].sum() > 0:
                    fig_gex.add_trace(go.Scatter(x=gex_df['Strike'], y=gex_df['VGEX_Call'], name='Call V-GEX (Flow)', mode='lines', line=dict(color='#00d2d3', width=3, dash='dash')))
                    fig_gex.add_trace(go.Scatter(x=gex_df['Strike'], y=gex_df['VGEX_Put'], name='Put V-GEX (Flow)', mode='lines', line=dict(color='#ff9ff3', width=3, dash='dash')))
                    fig_gex.add_trace(go.Scatter(x=gex_df['Strike'], y=gex_df['Net_VGEX'], name='Net V-GEX', mode='lines', line=dict(color='white', width=3)))

                if simulated_shift == 0:
                    fig_gex.add_vline(x=current_price, line_dash="dot", line_color="white", annotation_text="Current Price")
                else:
                    fig_gex.add_vline(x=current_price, line_dash="solid", line_color="gray", annotation_text="Original Spot", opacity=0.5)
                    fig_gex.add_vline(x=simulated_price, line_dash="dot", line_color="red", annotation_text=f"Simulated Spot ({simulated_shift:+.1f}%)")
                    
                fig_gex.update_layout(barmode='relative', xaxis_title="Strike Price", yaxis_title="Gamma Exposure", hovermode="x unified", height=500)
                st.plotly_chart(fig_gex, use_container_width=True)
                
                # ==========================================
                # 🎯 Tactical Execution: Theta-Optimized Spreads
                # ==========================================
                st.subheader("🎯 Tactical Execution: Theta-Optimized Credit Spreads")
                st.caption("สแกนหาจุดตั้งรับ (Credit Spread) นอกเขต Gamma Wall และ Support Floor ที่ได้เปรียบด้านเวลา (Theta Decay)")
                
                bear_call_spread, bull_put_spread = find_optimal_credit_spreads(simulated_price, gamma_wall, support_floor, gex_df)
                
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    if bear_call_spread:
                        st.info(f"**📉 Bear Call Spread (ต้านทานขาขึ้น)**\n\n"
                                f"• **Short Strike**: {bear_call_spread['short']} (ขาย Premium ขอบกำแพง)\n"
                                f"• **Long Strike**: {bear_call_spread['long']} (ซื้อประกันความเสี่ยง)\n"
                                f"• **Theta Edge**: {bear_call_spread['theta_edge']:.3f}")
                    else:
                        st.info("📉 ยังไม่มีจังหวะทำ Bear Call Spread ที่ได้เปรียบ")
                        
                with tcol2:
                    if bull_put_spread:
                        st.success(f"**📈 Bull Put Spread (รับมือขาลง)**\n\n"
                                   f"• **Short Strike**: {bull_put_spread['short']} (รับซื้อที่แนว Support)\n"
                                   f"• **Long Strike**: {bull_put_spread['long']} (ซื้อประกันความเสี่ยง)\n"
                                   f"• **Theta Edge**: {bull_put_spread['theta_edge'] if isinstance(bull_put_spread['theta_edge'], str) else format(bull_put_spread['theta_edge'], '.3f')}")
                    else:
                        st.success("📈 ยังไม่มีจังหวะทำ Bull Put Spread ที่ได้เปรียบ")
                
            else:
                 gex_df = pd.DataFrame()
                 spi_score = 0.0
                 skew_ratio = 1.0
            
            col1, col2, col3 = st.columns(3)
            # 2️⃣ กราฟ OI
            with col1:
                st.subheader("📊 Open Interest")
                fig_oi = px.bar(barmode="group")
                fig_oi.add_bar(x=df_call["Strike"], y=df_call["Value"], name="Call OI", marker_color="#2ecc71")
                fig_oi.add_bar(x=df_put["Strike"], y=df_put["Value"], name="Put OI", marker_color="#e74c3c")
                if current_price > 0: fig_oi.add_vline(x=current_price, line_dash="dash", line_color="white")
                fig_oi.update_layout(xaxis_title="Strike Price", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_oi, use_container_width=True)

            # 🆕 กราฟ Volume (Intraday)
            with col2:
                st.subheader("📢 Intraday Volume")
                if not df_vol_call.empty and not df_vol_put.empty:
                    fig_vol = px.bar(barmode="group")
                    fig_vol.add_bar(x=df_vol_call["Strike"], y=df_vol_call["Value"], name="Call Vol", marker_color="#00b894")
                    fig_vol.add_bar(x=df_vol_put["Strike"], y=df_vol_put["Value"], name="Put Vol", marker_color="#d63031")
                    if current_price > 0: fig_vol.add_vline(x=current_price, line_dash="dash", line_color="white")
                    fig_vol.update_layout(xaxis_title="Strike Price", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    st.info("⚠️ ไม่พบข้อมูล Volume ระหว่างวัน")

            # 3️⃣ กราฟ Max Pain
            with col3:
                pain_df, max_pain_strike = calculate_max_pain(df_call, df_put)
                st.subheader("🧲 Max Pain")
                st.markdown(f"**Target:** <span style='color:#f1c40f; font-size:18px;'>{max_pain_strike}</span>", unsafe_allow_html=True)
                fig_pain = px.line(pain_df, x="Strike", y="Total Pain", color_discrete_sequence=["yellow"])
                fig_pain.add_vline(x=max_pain_strike, line_dash="dash", line_color="#f1c40f")
                if current_price > 0: fig_pain.add_vline(x=current_price, line_dash="dot", line_color="white")
                fig_pain.update_layout(xaxis_title="Strike Price", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_pain, use_container_width=True)

            # ==========================================
            # 4️⃣ กราฟ IV แยก Call/Put Side
            # ==========================================
            st.markdown("---")
            st.subheader("📉 Implied Volatility Analysis")
            
            eod_times = [t for t in available_times if t.endswith("_EOD")]
            vol_df = pd.DataFrame()
            
            if eod_times:
                latest_eod = sorted(eod_times)[-1]
                eod_raw = data[selected_date][selected_series][latest_eod]
                
                if eod_raw.get("points"):
                    eod_df = pd.DataFrame(eod_raw["points"])
                    vol_df = eod_df[eod_df["DataType"] == "Volatility"].copy()
                    vol_df = vol_df.sort_values("Strike").reset_index(drop=True)
                    
                    if not vol_df.empty and current_price > 0:
                        # แบ่ง Put-side (OTM Put = below ATM) vs Call-side (OTM Call = above ATM)
                        vol_df["Side"] = vol_df["Strike"].apply(
                            lambda k: "Put-side" if k < current_price else "Call-side"
                        )
                        
                        put_side = vol_df[vol_df["Strike"] < current_price]
                        call_side = vol_df[vol_df["Strike"] >= current_price]
                        
                        avg_put_iv = put_side["Value"].mean() * 100 if not put_side.empty else 0
                        avg_call_iv = call_side["Value"].mean() * 100 if not call_side.empty else 0
                        atm_iv = vol_df.iloc[(vol_df["Strike"] - current_price).abs().argsort().iloc[0]]["Value"] * 100
                        skew_ratio = avg_put_iv / avg_call_iv if avg_call_iv > 0 else 0
                        
                        # 📊 Metric boxes
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("🎯 ATM IV", f"{atm_iv:.2f}%")
                        m2.metric("🔴 Avg Put IV", f"{avg_put_iv:.2f}%")
                        m3.metric("🟢 Avg Call IV", f"{avg_call_iv:.2f}%")
                        skew_label = "Bearish Skew" if skew_ratio > 1.05 else ("Bullish Skew" if skew_ratio < 0.95 else "Balanced")
                        m4.metric("⚖️ Put/Call Ratio", f"{skew_ratio:.3f}", delta=skew_label, delta_color="inverse" if skew_ratio > 1 else "normal")
                        
                        st.caption(f"ข้อมูลล่าสุด: {latest_eod} | Put/Call IV Ratio > 1 = ตลาดกลัวขาลง (Bearish Skew)")
                        
                        # 📈 กราฟ IV แยก Call/Put
                        fig_iv = go.Figure()
                        
                        if not put_side.empty:
                            fig_iv.add_trace(go.Scatter(
                                x=put_side["Strike"], y=put_side["Value"] * 100,
                                name="Put-side IV", mode="lines+markers",
                                line=dict(color="#e74c3c", width=2),
                                fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.1)"
                            ))
                        if not call_side.empty:
                            fig_iv.add_trace(go.Scatter(
                                x=call_side["Strike"], y=call_side["Value"] * 100,
                                name="Call-side IV", mode="lines+markers",
                                line=dict(color="#2ecc71", width=2),
                                fill="tozeroy", fillcolor="rgba(46, 204, 113, 0.1)"
                            ))
                        
                        fig_iv.add_vline(x=current_price, line_dash="dot", line_color="white", annotation_text=f"ATM ({current_price})")
                        fig_iv.add_hline(y=atm_iv, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text=f"ATM IV: {atm_iv:.2f}%")
                        
                        fig_iv.update_layout(
                            xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
                            hovermode="x unified", height=450, 
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_iv, use_container_width=True)
                    elif not vol_df.empty:
                        fig_vol_basic = px.line(vol_df, x="Strike", y="Value", markers=True)
                        fig_vol_basic.update_layout(xaxis_title="Strike Price", yaxis_title="IV (%)", hovermode="x unified", height=400)
                        st.plotly_chart(fig_vol_basic, use_container_width=True)
                    else:
                        st.info(f"ℹ️ พบข้อมูล EOD ({latest_eod}) แต่ไม่พบค่า Volatility")
                else:
                    st.warning(f"⚠️ ข้อมูล EOD ({latest_eod}) ว่างเปล่า")
            else:
                st.warning(f"⚠️ ไม่พบข้อมูล Volatility (EOD) สำหรับ {selected_series} ในวันนี้")

            # ==========================================
            # 5️⃣ เปรียบเทียบ IV ข้ามวัน (Cross-Day)
            # ==========================================
            st.markdown("---")
            st.subheader("📊 IV Cross-Day Comparison")
            
            all_dates_sorted = sorted(data.keys(), reverse=True)
            
            if len(all_dates_sorted) >= 2 and not vol_df.empty:
                # หาวันก่อนหน้าที่มีข้อมูล
                prev_vol_df = pd.DataFrame()
                prev_date_used = None
                prev_series_used = None
                
                for prev_date in all_dates_sorted:
                    if prev_date == selected_date:
                        continue
                    
                    # หา series เดียวกันหรือ series ที่คล้ายกันในวันก่อนหน้า
                    if selected_series in data.get(prev_date, {}):
                        prev_series_data = data[prev_date][selected_series]
                        prev_series_used = selected_series
                    else:
                        # ลองหา series อื่นในวันนั้น
                        available_prev = list(data.get(prev_date, {}).keys())
                        if not available_prev:
                            continue
                        prev_series_data = data[prev_date][available_prev[0]]
                        prev_series_used = available_prev[0]
                    
                    # หา EOD data ของวันก่อนหน้า
                    prev_eod_times = [t for t in prev_series_data.keys() if t.endswith("_EOD")]
                    if prev_eod_times:
                        latest_prev_eod = sorted(prev_eod_times)[-1]
                        prev_eod_raw = prev_series_data[latest_prev_eod]
                        
                        if prev_eod_raw.get("points"):
                            prev_eod_df = pd.DataFrame(prev_eod_raw["points"])
                            prev_vol_df = prev_eod_df[prev_eod_df["DataType"] == "Volatility"].copy()
                            prev_vol_df = prev_vol_df.sort_values("Strike").reset_index(drop=True)
                            prev_date_used = prev_date
                            break
                
                if not prev_vol_df.empty and prev_date_used:
                    st.caption(f"📅 เปรียบเทียบ: **{selected_date}** ({selected_series}) vs **{prev_date_used}** ({prev_series_used})")
                    
                    col_overlay, col_change = st.columns(2)
                    
                    with col_overlay:
                        st.markdown("**🔀 IV Overlay (ซ้อนทับ)**")
                        fig_compare = go.Figure()
                        
                        fig_compare.add_trace(go.Scatter(
                            x=vol_df["Strike"], y=vol_df["Value"] * 100,
                            name=f"{selected_date}", mode="lines+markers",
                            line=dict(color="#f1c40f", width=3)
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=prev_vol_df["Strike"], y=prev_vol_df["Value"] * 100,
                            name=f"{prev_date_used}", mode="lines",
                            line=dict(color="#95a5a6", width=2, dash="dash")
                        ))
                        
                        if current_price > 0:
                            fig_compare.add_vline(x=current_price, line_dash="dot", line_color="white", opacity=0.5)
                        
                        fig_compare.update_layout(
                            xaxis_title="Strike Price", yaxis_title="IV (%)",
                            hovermode="x unified", height=400,
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
                    with col_change:
                        st.markdown("**📈 IV Change (ส่วนต่าง)**")
                        # Merge on Strike เพื่อคำนวณ IV Change
                        merged = pd.merge(
                            vol_df[["Strike", "Value"]].rename(columns={"Value": "IV_Today"}),
                            prev_vol_df[["Strike", "Value"]].rename(columns={"Value": "IV_Prev"}),
                            on="Strike", how="inner"
                        )
                        
                        if not merged.empty:
                            merged["IV_Change"] = (merged["IV_Today"] - merged["IV_Prev"]) * 100
                            
                            fig_change = go.Figure()
                            colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in merged["IV_Change"]]
                            
                            fig_change.add_trace(go.Bar(
                                x=merged["Strike"], y=merged["IV_Change"],
                                marker_color=colors, name="IV Change"
                            ))
                            
                            fig_change.add_hline(y=0, line_color="white", line_width=1)
                            if current_price > 0:
                                fig_change.add_vline(x=current_price, line_dash="dot", line_color="white", opacity=0.5)
                            
                            fig_change.update_layout(
                                xaxis_title="Strike Price", yaxis_title="IV Change (pp)",
                                hovermode="x unified", height=400
                            )
                            st.plotly_chart(fig_change, use_container_width=True)
                            
                            # สรุป IV Change
                            avg_change = merged["IV_Change"].mean()
                            max_increase = merged.loc[merged["IV_Change"].idxmax()]
                            max_decrease = merged.loc[merged["IV_Change"].idxmin()]
                            
                            change_icon = "🔺" if avg_change > 0 else "🔻"
                            st.caption(f"{change_icon} **Avg IV Change: {avg_change:+.3f}pp** | "
                                      f"📈 Max increase: Strike {max_increase['Strike']:.0f} ({max_increase['IV_Change']:+.3f}pp) | "
                                      f"📉 Max decrease: Strike {max_decrease['Strike']:.0f} ({max_decrease['IV_Change']:+.3f}pp)")
                        else:
                            st.info("⚠️ ไม่มี Strike ที่ตรงกันระหว่างสองวัน")
                else:
                    st.info("ℹ️ ไม่พบข้อมูล Volatility ของวันก่อนหน้าสำหรับเปรียบเทียบ")
            elif vol_df.empty:
                st.info("ℹ️ ไม่มีข้อมูล IV วันนี้สำหรับเปรียบเทียบ")
            else:
                st.info("ℹ️ ต้องมีข้อมูลอย่างน้อย 2 วันเพื่อเปรียบเทียบ")

            # ==========================================
            # 🧊 3D Volatility Term Structure Surface
            # ==========================================
            st.markdown("---")
            st.subheader("🧊 3D Volatility Term Structure Surface")
            st.caption("โครงสร้างของความผันผวนข้ามเวลา (Term Structure) ชี้เป้าความผิดปกติของราคา (Anomalies)")
            
            x_strikes = []
            y_dtes = []
            z_ivs = []
            
            for s_name, s_data in data[selected_date].items():
                eod_pts = [t for t in s_data.keys() if t.endswith("_EOD")]
                if eod_pts:
                    let_latest_eod = sorted(eod_pts)[-1]
                    s_raw = s_data[let_latest_eod]
                    if s_raw.get("points") and s_raw.get("dte"):
                        df_s = pd.DataFrame(s_raw["points"])
                        df_vol_s = df_s[df_s["DataType"] == "Volatility"].copy()
                        if not df_vol_s.empty:
                            s_dte = s_raw["dte"]
                            # Only keep reasonable strikes around current_price to cut outliers
                            df_vol_s = df_vol_s[(df_vol_s["Strike"] >= current_price * 0.8) & (df_vol_s["Strike"] <= current_price * 1.2)]
                            
                            x_strikes.extend(df_vol_s["Strike"].tolist())
                            y_dtes.extend([s_dte] * len(df_vol_s))
                            z_ivs.extend((df_vol_s["Value"] * 100).tolist())
            
            if len(set(y_dtes)) >= 2: # At least 2 different expirations to make a surface
                fig_3d = go.Figure(data=[go.Mesh3d(
                    x=x_strikes,
                    y=y_dtes,
                    z=z_ivs,
                    opacity=0.8,
                    intensity=z_ivs,
                    colorscale='Viridis',
                    hovertemplate='Strike: %{x:.0f}<br>DTE: %{y} Days<br>IV: %{z:.2f}%<extra></extra>',
                    showscale=True
                )])
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title='Strike Price',
                        yaxis_title='Days to Expiration (DTE)',
                        zaxis_title='Implied Volatility (%)',
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
                    ),
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("ℹ️ ต้องการข้อมูลจากอย่างน้อย 2 วันหมดอายุที่มี Volatility เพื่อวาดกราฟ 3D Term Structure แบบเต็มรูปแบบ")

            # ==========================================
            # 🤖 พื้นที่ให้ AI แสดงฝีมือ (UI)
            # ==========================================
            st.markdown("---")
            st.subheader("🤖 AI Senior Quant Analyst")

            st.markdown(f"> **Model**: Groq - {GROQ_MODEL} (Exclusively Selected)")

            if st.button("✨ กดเพื่อให้ AI วิเคราะห์แผนการเทรด"):
                with st.spinner(f"🧠 AI ({GROQ_MODEL}) กำลังวิเคราะห์... (Powered by Groq)"):
                    if GROQ_API_KEY == "NO_API_KEY_FOUND_IN_SECRETS":
                        st.error("❌ ไม่พบ API Key! กรุณาเพิ่ม GROQ_API_KEY ใน Streamlit Cloud Secrets")
                    else:
                        ai_insight = get_ai_analysis(
                            current_price=current_price, 
                            max_pain=max_pain_strike, 
                            gex_df=gex_df, 
                            dte=dte, 
                            series_name=f"{selected_series} (Exp: {selected_date})", 
                            vol_call_df=df_vol_call, 
                            vol_put_df=df_vol_put,
                            spi_score=spi_score,
                            skew_ratio=skew_ratio_for_ai,
                            bear_call=bear_call_spread,
                            bull_put=bull_put_spread,
                            simulated_shift=simulated_shift,
                            simulated_price=simulated_price
                        )
                        st.success("✅ วิเคราะห์เสร็จสิ้น!")
                        st.markdown(ai_insight, unsafe_allow_html=True)

        else:
            st.warning("⚠️ ไม่พบข้อมูล Open Interest ในรอบนี้")
    else:
        st.error("❌ ไม่สามารถดึงข้อมูลจาก Firebase ได้")
