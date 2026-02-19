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
            # 🤖 พื้นที่ให้ AI แสดงฝีมือ (UI)
            # ==========================================
            st.markdown("---")
            st.subheader("🤖 AI Senior Quant Analyst")

            st.markdown(f"> **Model**: Groq - {GROQ_MODEL} (Exclusively Selected)")

            if st.button("✨ กดเพื่อให้ AI วิเคราะห์แผนการเทรด"):
                with st.spinner(f"🧠 AI ({GROQ_MODEL}) กำลังวิเคราะห์... (Powered by Groq)"):
                    ai_insight = get_ai_analysis(
                        current_price, max_pain_strike, gex_df, dte, selected_series, 
                        df_vol_call, df_vol_put
                    )
                    st.success("✅ วิเคราะห์เสร็จสิ้น!")
                    st.markdown(ai_insight, unsafe_allow_html=True)

        else:
            st.warning("⚠️ ไม่พบข้อมูล Open Interest ในรอบนี้")
    else:
        st.error("❌ ไม่สามารถดึงข้อมูลจาก Firebase ได้")
