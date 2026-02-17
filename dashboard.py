import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import subprocess
import os
import time

# ⚙️ ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="CME Quant Dashboard", layout="wide")

# 🛠️ ตรวจสอบและติดตั้ง Playwright Browser (สำหรับ Cloud Deploy)
try:
    # ลองเช็คว่ามี browser folder หรือยัง (แบบบ้านๆ) หรือสั่ง install เลยก็ได้
    # เพื่อความชัวร์ สั่ง install แบบ dry-run หรือ check ก่อนดีกว่า แต่ใน Streamlit Cloud สั่ง install blind ไปเลยมักจะ work กว่า
    # แต่อาจจะช้าตอน start ครั้งแรก
    if not os.path.exists("playwright_installed.flag"):
        with st.spinner("🔧 กำลังติดตั้ง Browser สำหรับ Bot... (ครั้งแรกเท่านั้น)"):
            subprocess.run(["playwright", "install", "chromium"], check=True)
            # สร้าง flag file
            with open("playwright_installed.flag", "w") as f:
                f.write("installed")
            print("✅ Playwright Browser Installed!")
except Exception as e:
    print(f"⚠️ Playwright Install Error: {e}")

st.title("🎯 CME Options Dashboard & AI Analysis")
st.markdown("ศูนย์บัญชาการวิเคราะห์โครงสร้างตลาด (OI, Max Pain, Net GEX พร้อม AI Quant Analyst)")

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

# ==========================================
# 🧠 ฟังก์ชันคณิตศาสตร์: Black-Scholes & GEX
# ==========================================
def calculate_gamma(F, K, T, sigma):
    if T <= 0 or sigma <= 0 or F <= 0: return 0.0
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

# ==========================================
# 🤖 ฟังก์ชัน AI Quant Analyst (Custom Prompt)
# ==========================================
def get_ai_analysis(current_price, max_pain, gex_df, dte, series_name, vol_call_df=None, vol_put_df=None):
    try:
        strongest_resistance = gex_df.loc[gex_df['GEX_Call'].idxmax()]['Strike']
        strongest_support = gex_df.loc[gex_df['GEX_Put'].idxmin()]['Strike']
    except:
        strongest_resistance = "N/A"
        strongest_support = "N/A"
    
    # ... (Volume Analysis Logic remains same) ...
    vol_analysis_txt = "ไม่มีข้อมูล Intraday Volume"
    if vol_call_df is not None and not vol_call_df.empty:
        try:
            top_call_vol = vol_call_df.nlargest(3, 'Value')[['Strike', 'Value']].to_dict('records')
            top_put_vol = vol_put_df.nlargest(3, 'Value')[['Strike', 'Value']].to_dict('records')
            vol_analysis_txt = f"Top Call Vol: {top_call_vol}, Top Put Vol: {top_put_vol}"
        except:
            vol_analysis_txt = "Error extracting volume data"
        
    prompt = f"""
    บทบาทและตรรกะหลัก (Role & Core Logic):
    * ทำหน้าที่เป็น 'Senior Quantitative Market Maker & Volatility Strategist' โดยยึดปรัชญา 'Risk is measured, not guessed.'
    * วิเคราะห์ตลาดผ่านโครงสร้าง Liquidity, Gamma Exposure (GEX), และ Volatility Surface โดยใช้หลักการ Chain-of-Thought (CoT) จากระดับ Micro Structure ไปจนถึง Macro Strategy.
    * ผสานความเชี่ยวชาญในฐานะ 'นัก Finance Engineering', 'นัก Prompt Engineering' และ 'Market Maker' เพื่อวิเคราะห์และพัฒนาโครงสร้างคำสั่ง (Prompt) พร้อมทั้งต่อยอดข้อมูลเชิงลึกจากการตลาด
    ... (Prompt content is preserved) ...
    =========================================
    📊 ข้อมูลอินพุตสำหรับวิเคราะห์รอบนี้ (Live Market Data):
    - Asset & Context: สัญญา {series_name}, DTE = {dte} วัน
    - Price Action: ราคาปัจจุบัน (Spot) = {current_price}
    - Intraday Volume Flow: {vol_analysis_txt}
    - Option Chain Data Summary: 
      * จุด Max Pain = {max_pain}
      * Gamma Wall (แนวต้าน Call GEX สูงสุด) = {strongest_resistance}
      * Support Floor (แนวรับ Put GEX ต่ำสุด) = {strongest_support}
    =========================================
    
    กรุณาดำเนินการวิเคราะห์ตามรูปแบบ Output Format ที่กำหนดไว้ทันที
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
# 📊 หน้าจอ Dashboard
# ==========================================
data = fetch_data()

if data:
    # ==========================================
    # 🚀 System Control (Remote Bot Trigger)
    # ==========================================
    with st.sidebar.expander("⚙️ System Control", expanded=True):
        if st.button("🚀 ดึงข้อมูลล่าสุด (Run Bot)", use_container_width=True):
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            
            try:
                with st.spinner("🤖 กำลังสั่ง Bot ดึงข้อมูล... (อาจใช้เวลา 1-2 นาที)"):
                    # Run bot script
                    process = subprocess.Popen(
                        ["python", "cme_bot.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    
                    stdout, stderr = process.communicate()
                    
                    if process.returncode == 0:
                        status_placeholder.success("✅ ดึงข้อมูลสำเร็จ! กำลังรีโหลด...")
                        with log_placeholder.expander("ดู Log การทำงาน", expanded=False):
                            st.code(stdout)
                        time.sleep(2)
                        st.rerun()
                    else:
                        status_placeholder.error("❌ เกิดข้อผิดพลาด")
                        with log_placeholder.expander("Error Log", expanded=True):
                            st.error(stderr)
                            st.code(stdout)
            except Exception as e:
                st.error(f"❌ Error running bot: {e}")

    st.sidebar.success("✅ เชื่อมต่อ Firebase สำเร็จ!")
    available_dates = list(data.keys())
    selected_date = st.sidebar.selectbox("📅 เลือกวันที่", sorted(available_dates, reverse=True))
    
    available_series = list(data[selected_date].keys())
    
    # 🛠️ สร้าง Map สำหรับแสดงชื่อ Series พร้อมวันหมดอายุ + สถานะ Active/Expired
    series_display_map = {}
    series_expiration_info = {}  # เก็บข้อมูล expiration สำหรับแสดงผลทีหลัง
    now = datetime.now()
    
    for s in available_series:
        try:
            first_time_key = list(data[selected_date][s].keys())[0]
            dte = data[selected_date][s][first_time_key].get('dte', 0)
            
            selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
            expiration_dt = selected_dt + timedelta(days=dte)
            exp_date = expiration_dt
            exp_str = expiration_dt.strftime("%d %b %Y")
            
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
            # Logic: เลือกตัวล่าสุดเสมอ (Simplest) หรือจะ Match เวลาที่ใกล้เคียงก็ได้
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
            # Filter DataType = Intraday และแยก Call/Put
            df_vol_call = df_intraday[(df_intraday["DataType"] == "Intraday") & (df_intraday["SeriesName"].str.contains("Call", case=False))]
            df_vol_put = df_intraday[(df_intraday["DataType"] == "Intraday") & (df_intraday["SeriesName"].str.contains("Put", case=False))]

        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ สภาวะตลาด (Parameters)**")
        st.sidebar.info(f"💵 ราคาปัจจุบัน: **{current_price}**\n\n⏳ วันหมดอายุ: **{dte} วัน**\n\n📈 ความผันผวน: **{iv:.4f}**")

        if not df_call.empty and not df_put.empty:
            
            # 1️⃣ Net GEX 
            st.subheader("🌋 Net Gamma Exposure (GEX) Profile")
            if current_price > 0 and dte > 0 and iv > 0:
                gex_df = process_gex(df_call, df_put, current_price, dte, iv)
                
                fig_gex = go.Figure()
                fig_gex.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX_Call'], name='Call GEX (+)', marker_color='rgba(46, 204, 113, 0.7)'))
                fig_gex.add_trace(go.Bar(x=gex_df['Strike'], y=gex_df['GEX_Put'], name='Put GEX (-)', marker_color='rgba(231, 76, 60, 0.7)'))
                fig_gex.add_trace(go.Scatter(x=gex_df['Strike'], y=gex_df['Net_GEX'], name='Net GEX', mode='lines+markers', line=dict(color='yellow', width=3)))
                
                fig_gex.add_vline(x=current_price, line_dash="dot", line_color="white", annotation_text="Current Price")
                fig_gex.update_layout(barmode='relative', xaxis_title="Strike Price", yaxis_title="Gamma Exposure", hovermode="x unified", height=450)
                st.plotly_chart(fig_gex, use_container_width=True)
            else:
                 gex_df = pd.DataFrame()
            
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
            # 4️⃣ กราฟ Volatility Skew (EOD)
            # ==========================================
            st.markdown("---")
            st.subheader("📉 Volatility Skew (EOD)")
            
            eod_times = [t for t in available_times if t.endswith("_EOD")]
            if eod_times:
                # เลือกตัวล่าสุด
                latest_eod = sorted(eod_times)[-1]
                eod_raw = data[selected_date][selected_series][latest_eod]
                
                if eod_raw.get("points"):
                    eod_df = pd.DataFrame(eod_raw["points"])
                    # Filter 'Volatility' OR SeriesName 'Vol Settle' (Fallback fix for old data)
                    vol_df = eod_df[
                        (eod_df["DataType"] == "Volatility") | 
                        (eod_df["SeriesName"].str.contains("Vol", case=False)) |
                        (eod_df["SeriesName"].str.contains("Settle", case=False))
                    ]
                    
                    if not vol_df.empty:
                        st.markdown(f"กราฟยิ้ม (Smile) หรือแสยะยิ้ม (Smirk) บอกว่าตลาดกลัวฝั่งไหนมากกว่ากัน (ข้อมูลล่าสุด: {latest_eod})")
                        
                        fig_vol = px.line(vol_df, x="Strike", y="Value", markers=True)
                        if current_price > 0: fig_vol.add_vline(x=current_price, line_dash="dot", line_color="white", annotation_text="Spot")
                        
                        fig_vol.update_layout(xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)", hovermode="x unified", height=400)
                        st.plotly_chart(fig_vol, use_container_width=True)
                    else:
                        st.info(f"ℹ️ พบข้อมูล EOD ({latest_eod}) แต่ไม่พบค่า Volatility ในข้อมูล")
                else:
                    st.warning(f"⚠️ ข้อมูล EOD ({latest_eod}) ว่างเปล่า")
            else:
                st.warning(f"⚠️ ไม่พบข้อมูล Volatility (EOD) สำหรับ {selected_series} ในวันนี้ (บอทอาจจะยังไม่ได้ดึง หรือดึงไม่สำเร็จ)")

            # ==========================================
            # 🤖 พื้นที่ให้ AI แสดงฝีมือ (UI)
            # ==========================================
            st.markdown("---")
            st.subheader("🤖 AI Senior Quant Analyst")

            # แสดง Info ว่าใช้โมเดลอะไรอยู่
            st.markdown(f"> **Model**: Groq - {GROQ_MODEL} (Exclusively Selected)")

            if st.button("✨ กดเพื่อให้ AI วิเคราะห์แผนการเทรด"):
                with st.spinner(f"🧠 AI ({GROQ_MODEL}) กำลังวิเคราะห์... (Powered by Groq)"):
                    ai_insight = get_ai_analysis(
                        current_price, max_pain_strike, gex_df, dte, selected_series, 
                        df_vol_call, df_vol_put
                    )
                    st.success("✅ วิเคราะห์เสร็จสิ้น!")
                    
                    # ใช้ st.markdown เพื่อรองรับการเรนเดอร์สมการ LaTeX และ Emoji จาก Prompt
                    st.markdown(ai_insight, unsafe_allow_html=True)

        else:
            st.warning("⚠️ ไม่พบข้อมูล Open Interest ในรอบนี้")
    else:
        st.error("❌ ไม่สามารถดึงข้อมูลจาก Firebase ได้")