from playwright.sync_api import sync_playwright
import requests
import time
from datetime import datetime, timedelta

# 🌟 URL ของ Firebase ของคุณ
FIREBASE_URL_TEMPLATE = "https://cme-quant-bot-default-rtdb.asia-southeast1.firebasedatabase.app/daily_data/{date}/{contract}/{time}_{type}.json"

# =========================================================
# ⏰ Expiration Time Utilities (Thai Time / CME CT)
# CME Gold Options หยุดเทรด 12:30 PM CT
# CT (Winter/CST, UTC-6): 12:30 PM → 01:30 AM Thai (วันถัดไป)
# CDT (Summer/CDT, UTC-5): 12:30 PM → 00:30 AM Thai (วันถัดไป)
# =========================================================

def is_us_dst(dt):
    """ตรวจว่าวันที่กำหนดอยู่ในช่วง US Daylight Saving Time หรือไม่
    DST: 2nd Sunday of March → 1st Sunday of November"""
    year = dt.year
    # หา 2nd Sunday of March
    mar1 = datetime(year, 3, 1)
    days_to_sun = (6 - mar1.weekday()) % 7
    dst_start = mar1 + timedelta(days=days_to_sun + 7)  # 2nd Sunday
    
    # หา 1st Sunday of November
    nov1 = datetime(year, 11, 1)
    days_to_sun = (6 - nov1.weekday()) % 7
    dst_end = nov1 + timedelta(days=days_to_sun)  # 1st Sunday
    
    return dst_start <= dt.replace(hour=0, minute=0, second=0, microsecond=0) < dst_end

def get_expiration_datetime_thai(exp_date):
    """คำนวณเวลาหมดอายุจริงเป็น Thai local time
    - exp_date: datetime ของวันหมดอายุ (จาก CME dropdown)
    - return: datetime ที่แม่นยำ (Thai time) ที่ series จะหมดอายุ
    
    CME Gold Options close: 12:30 PM CT
    Winter (CST, UTC-6): +13h30m → 01:30 AM Thai time (day after exp_date)
    Summer (CDT, UTC-5): +12h30m → 00:30 AM Thai time (day after exp_date)
    """
    if is_us_dst(exp_date):
        # CDT: 12:30 PM + 12h30m = 00:30 AM next day Thai
        return exp_date + timedelta(hours=24, minutes=30)  # = exp_date+1 day, 00:30
    else:
        # CST: 12:30 PM + 13h30m = 01:30 AM next day Thai
        return exp_date + timedelta(days=1, hours=1, minutes=30)  # = exp_date+1 day, 01:30

def is_series_active(exp_date, now=None):
    """ตรวจว่า series ยังไม่หมดอายุ ณ เวลาปัจจุบัน (Thai time)"""
    if now is None:
        now = datetime.now()
    expiration_cutoff = get_expiration_datetime_thai(exp_date)
    return now < expiration_cutoff

def get_highcharts_data(page):
    """ฟังก์ชันแอบล้วงข้อมูลจากกราฟ"""
    return page.evaluate("""() => {
        if (window.Highcharts && window.Highcharts.charts) {
            let chart = window.Highcharts.charts.find(c => c !== undefined);
            if (chart && chart.options) {
                let rawTitle = chart.title ? chart.title.textStr : "Unknown Contract";
                let cleanTitle = rawTitle.replace(/<[^>]*>?/gm, '').replace(/&nbsp;/g, ' ').trim();
                let pureContract = cleanTitle.split(' ')[0];
                let settings = chart.userOptions.custom ? chart.userOptions.custom.Settings : {};
                
                let extractedPoints = [];
                chart.series.forEach(series => {
                    if(series.data && series.data.length > 0) {
                        let sName = series.name || "Unknown";
                        let sId = series.options.id || "";
                        let yTitle = (series.yAxis && series.yAxis.axisTitle && series.yAxis.axisTitle.textStr) ? series.yAxis.axisTitle.textStr : "";
                        
                        let dataType = "-";
                        let checkStr = (sName + " " + sId + " " + yTitle).toLowerCase();
                        if (checkStr.includes("oi") || checkStr.includes("open interest")) dataType = "OI";
                        else if (sName.includes("Settle") || checkStr.includes("volatility")) dataType = "Volatility";
                        else if (checkStr.includes("vol") || checkStr.includes("intraday")) dataType = "Intraday";
                        else if (sName === "Ranges") dataType = "Probability Band";
                        
                        series.data.forEach(point => {
                            if(point.x !== undefined && point.y !== undefined) {
                                extractedPoints.push({
                                    Contract: pureContract, SeriesName: sName, DataType: dataType, Strike: point.x, Value: point.y
                                });
                            }
                        });
                    }
                });
                return { contract: pureContract, full_title: cleanTitle, price: settings.FuturePrice, dte: settings.DTE, vol_settle: settings.ATMVol, points: extractedPoints };
            }
        }
        return null;
    }""")

def push_to_firebase(data, data_type):
    """ส่งข้อมูลขึ้น Cloud อัตโนมัติ"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    url = FIREBASE_URL_TEMPLATE.format(date=date_str, contract=data['contract'], time=time_str, type=data_type)
    
    try:
        response = requests.put(url, json=data)
        if response.status_code == 200:
            print(f"✅ อัปโหลด {data['contract']} [{data_type}] ขึ้น Firebase สำเร็จ! ({time_str})")
        else:
            print(f"❌ Error Uploading: {response.text}")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Firebase ได้: {e}")

# =========================================================
# 📦 ฟังก์ชันเดิม: เหมาจบ ดึง Intraday + OI
# =========================================================
def process_current_series(page, series_type_name):
    print(f"\n--- 🚀 เริ่มดูดข้อมูลชุด: {series_type_name} ---")
    
    # 1. รอ Intraday
    print(f"📥 [{series_type_name}] กำลังรอข้อมูล Intraday... (จะไม่ไปต่อจนกว่าจะเจอ)")
    
    intraday_data = None
    attempt = 1
    while True:
        intraday_data = get_highcharts_data(page)
        if intraday_data:
            print(f"✅ [{series_type_name}] เจอข้อมูล Intraday แล้ว! (รอบที่ {attempt})")
            push_to_firebase(intraday_data, "Intraday")
            break
        else:
            print(f"⏳ [{series_type_name}] ยังไม่เจอ Intraday... รออีก 3 วินาที (รอบที่ {attempt})")
            time.sleep(3)
            attempt += 1

    # 2. กดปุ่ม OI
    print(f"🖱️ [{series_type_name}] Intraday ครบแล้ว -> กำลังกดไปหา 'Open Interest'...")
    
    while True:
        try:
            target_button = None
            try:
                potential_btn = page.locator("a[id$='lbOI']").first
                if potential_btn.is_visible():
                     print(f"🎯 เจอโดย ID (lbOI): {potential_btn.get_attribute('id')}")
                     target_button = potential_btn
            except:
                pass

            if not target_button:
                print("⚠️ ไม่เจอ ID lbOI... ลองหาจาก Text")
                vtabs = page.locator("li.vtab").all()
                for vtab in vtabs:
                    classes = vtab.get_attribute("class") or ""
                    txt = vtab.inner_text().strip()
                    
                    if "spacer" in classes:
                        continue
                    
                    if txt == "OI" or "Open Interest" in txt:
                        print(f"  ✅ เจอ Tab น่าสนใจ: {txt} (Class: {classes})")
                        link = vtab.locator("a").first
                        if link.count() > 0:
                            target_button = link
                        else:
                            target_button = vtab
                        break

            if target_button:
                print(f"🎯 เลือกคลิกตัวนี้: {target_button.inner_text()} (Tag: {target_button.evaluate('el => el.tagName')})")
                target_button.click(force=True)
                print("👇 กดปุ่มแล้ว รอโหลด...")
                time.sleep(5) 
            else:
                print("❌ หาปุ่มที่น่าจะเป็น Clickable 'Open Interest' ไม่เจอ")

            current_data = get_highcharts_data(page)
            break 
            
        except Exception as e:
            print(f"❌ Error while clicking OI: {e} ... ลองใหม่ใน 3 วินาที")
            time.sleep(3)

    # 3. รอ OI
    print(f"📥 [{series_type_name}] กำลังรอข้อมูล OI...")
    oi_data = None
    attempt = 1
    while True:
        raw_data = get_highcharts_data(page)
        if raw_data:
             is_oi = False
             if raw_data.get('points'):
                 first_point = raw_data['points'][0]
                 if first_point.get('DataType') == 'OI':
                     is_oi = True
            
             if is_oi:
                print(f"✅ [{series_type_name}] เจอข้อมูล OI แล้ว! (รอบที่ {attempt})")
                push_to_firebase(raw_data, "OI")
                oi_data = raw_data # เก็บไว้ return
                break
             else:
                title = raw_data.get('full_title', 'No Title')
                first_point_type = raw_data['points'][0].get('DataType') if raw_data.get('points') else "No Points"
                print(f"⏳ [{series_type_name}] ข้อมูลที่ได้ยังไม่ใช่ OI (Type: {first_point_type}, Title: {title})... รอโหลดใหม่ (รอบที่ {attempt})")
                
                if attempt % 5 == 0:
                    print(f"⚠️ รอนานผิดปกติ (รอบที่ {attempt}) -> ลองกดปุ่ม 'Open Interest' ย้ำอีกที...")
                    try:
                        page.locator("text='Open Interest' >> visible=true").first.click(force=True)
                    except:
                        pass
                
                time.sleep(3)
        else:
            print(f"⏳ [{series_type_name}] ยังไม่เจอกราฟ... รออีก 3 วินาที (รอบที่ {attempt})")
            time.sleep(3)
        
        attempt += 1
    
    # -------------------------------------
    # 3. 🎯 ภารกิจใหม่: สลับไปหน้า EOD (Volatility Surface)
    # -------------------------------------
    print(f"🖱️ [{series_type_name}] กำลังค้นหาและคลิกแท็บ 'EOD' (Volatility)...")
    try:
        # หาปุ่มที่มีคำว่า EOD แล้วบังคับคลิก
        eod_btn = page.locator("text='EOD' >> visible=true")
        if eod_btn.count() > 0:
            eod_btn.first.click(force=True)
            print(f"⏳ [{series_type_name}] รอให้กราฟ EOD โหลดข้อมูลใหม่ 5 วินาที...")
            time.sleep(5) 

            print(f"📥 [{series_type_name}] กำลังดูดข้อมูล EOD Volatility...")
            eod_data = get_highcharts_data(page)
            
            # โยนขึ้น Firebase พร้อมตั้งชื่อ Type ว่า 'EOD'
            if eod_data: 
                push_to_firebase(eod_data, "EOD")
            else:
                print(f"⚠️ [{series_type_name}] ไม่ได้ข้อมูล EOD (get_highcharts_data return None)")
        else:
            print(f"❌ [{series_type_name}] หาปุ่ม EOD ไม่เจอ!")
            
    except Exception as e:
        print(f"❌ [{series_type_name}] บอทคลิกเปลี่ยนหน้า EOD พลาด: {e}")
        # page.screenshot(path=f"error_screen_eod_{series_type_name}.png")

    print(f"--- ✅ จบภารกิจชุด: {series_type_name} (ครบถ้วน!) ---\n")
    
    extracted_contract = "Unknown"
    if intraday_data:
        extracted_contract = intraday_data.get('contract', 'Unknown')
    elif oi_data:
         extracted_contract = oi_data.get('contract', 'Unknown')
         
    return extracted_contract

# =========================================================
# 🧠 ฟังก์ชันฉลาด: เลือก Series รองที่ใกล้วันนี้ที่สุด
# (ใช้เวลาหมดอายุจริง ~ตี 1:30 Thai / 00:30 ช่วง DST)
# =========================================================
def select_best_secondary_series(page, current_main_series_code="OGJ6"):
    print("\n🔍 กำลังค้นหา Series รองที่เหมาะสมที่สุด...")
    
    options = page.locator(".link-selector ul.nav li a").all()
    
    candidates = []
    now = datetime.now()
    
    dst_status = "CDT (Summer)" if is_us_dst(now) else "CST (Winter)"
    print(f"📅 เวลาไทยตอนนี้: {now.strftime('%d %b %Y %H:%M:%S')} | US Timezone: {dst_status}")

    for opt in options:
        try:
            full_text = opt.inner_text().strip()
            lines = full_text.split('\n')
            
            if len(lines) >= 2:
                code = lines[0].strip()
                date_str = lines[1].strip()
                
                try:
                    exp_date = datetime.strptime(date_str, "%d %b %Y")
                    expiry_thai = get_expiration_datetime_thai(exp_date)
                    active = is_series_active(exp_date, now)
                    status_icon = "🟢" if active else "🔴"
                    
                    print(f"  {status_icon} {code} | Exp: {date_str} | Cutoff Thai: {expiry_thai.strftime('%d %b %H:%M')} | {'ACTIVE' if active else 'EXPIRED'}")
                    
                    # เอาเฉพาะ series ที่ยังไม่หมดอายุ และไม่ใช่ตัวหลัก
                    if active and code != current_main_series_code:
                        candidates.append({
                            'code': code,
                            'date': exp_date,
                            'expiry_thai': expiry_thai,
                            'element': opt
                        })
                except ValueError:
                    continue
        except:
            continue

    if candidates:
        # Sort by date ascending (ใกล้สุดก่อน)
        candidates.sort(key=lambda x: x['date'])
        best_choice = candidates[0]
        
        print(f"🎯 เจอเป้าหมาย! Series รองที่ดีที่สุดคือ: {best_choice['code']} (Exp: {best_choice['date'].strftime('%d %b %Y')}, Cutoff: {best_choice['expiry_thai'].strftime('%d %b %H:%M')})")
        
        best_choice['element'].click(force=True)
        return best_choice['code']
    else:
        print("⚠️ ไม่เจอ Series รองที่เหมาะสมเลย (ทุกตัวหมดอายุแล้ว หรือเหลือแค่ตัวหลัก)")
        return None


def run_bot():
    print("🚀 เริ่มต้นเดินเครื่อง Bot ล่องหน...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        print("🌐 กำลังเข้าเว็บ CME...")
        page.goto("https://cmegroup-tools.quikstrike.net//User/QuikStrikeView.aspx?pid=40&pf=6&viewitemid=IntegratedV2VExpectedRange&insid=209341331&qsid=baf9e29e-d0c4-4680-bac6-5b07eabe76b5")
        
        page.wait_for_load_state("domcontentloaded")
        print("⏳ รอให้ UI และกราฟเตรียมตัว 8 วินาที...")
        time.sleep(8) 

        # 🎯 ภารกิจที่ 1: ดึงซีรีส์หลัก
        main_contract = process_current_series(page, "Main_Monthly")
        print(f"📌 Main Contract detected: {main_contract}")

        # 🎯 ภารกิจที่ 2: เปลี่ยนซีรีส์แบบ Dynamic
        print("🔄 กำลังสลับไปหา 'ซีรีส์รอง' (เปลี่ยน Dropdown)...")
        try:
            page.locator("text='Expiration:' >> visible=true").first.click(force=True)
            time.sleep(2)
            
            selected_code = select_best_secondary_series(page, current_main_series_code=main_contract)
            
            if selected_code:
                print("⏳ รอให้กราฟซีรีส์รองโหลด 8 วินาที...")
                time.sleep(8)
                
                # กดกลับมาหน้า Intraday ก่อน
                try:
                    intraday_btn = page.locator("text='Intraday Volume' >> visible=true")
                    if intraday_btn.count() > 0:
                        intraday_btn.first.click(force=True)
                        time.sleep(5)
                except:
                    pass

                process_current_series(page, f"Secondary_{selected_code}")
            else:
                print("❌ ข้ามภารกิจที่ 2 เพราะเลือก Series ไม่ได้")

        except Exception as e:
            print(f"❌ เปลี่ยนซีรีส์ไม่สำเร็จ: {e}")
            page.screenshot(path="error_change_series.png")

        print("🎉 ดึงข้อมูลรอบนี้เสร็จสิ้น ปิดเบราว์เซอร์!")
        browser.close()

if __name__ == "__main__":
    run_bot()