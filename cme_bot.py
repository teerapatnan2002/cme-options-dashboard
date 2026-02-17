from playwright.sync_api import sync_playwright
import requests
import time
from datetime import datetime, timedelta

# 🌟 URL ของ Firebase ของคุณ
FIREBASE_URL_TEMPLATE = "https://cme-quant-bot-default-rtdb.asia-southeast1.firebasedatabase.app/daily_data/{date}/{contract}/{time}_{type}.json"

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
    expiration_cutoff = get_expiration_datetime_thai(exp_date)
    return now < expiration_cutoff

def get_highcharts_data(page):
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

def process_current_series(page, series_type_name):
    print(f"--- 🚀 เริ่มดูดข้อมูลชุด: {series_type_name} ---")
    intraday_data = None
    attempt = 1
    while True:
        intraday_data = get_highcharts_data(page)
        if intraday_data:
            push_to_firebase(intraday_data, "Intraday")
            break
        time.sleep(3)
        attempt += 1

    while True:
        try:
            target_button = None
            potential_btn = page.locator("a[id$='lbOI']").first
            if potential_btn.is_visible():
                 target_button = potential_btn
            if not target_button:
                vtabs = page.locator("li.vtab").all()
                for vtab in vtabs:
                    txt = vtab.inner_text().strip()
                    if txt == "OI" or "Open Interest" in txt:
                        link = vtab.locator("a").first
                        target_button = link if link.count() > 0 else vtab
                        break
            if target_button:
                target_button.click(force=True)
                time.sleep(5) 
            current_data = get_highcharts_data(page)
            break 
        except:
            time.sleep(3)

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
                push_to_firebase(raw_data, "OI")
                oi_data = raw_data
                break
             if attempt % 5 == 0:
                try: page.locator("text='Open Interest' >> visible=true").first.click(force=True)
                except: pass
             time.sleep(3)
        else:
            time.sleep(3)
        attempt += 1
    
    try:
        eod_btn = page.locator("text='EOD' >> visible=true")
        if eod_btn.count() > 0:
            eod_btn.first.click(force=True)
            time.sleep(5) 
            eod_data = get_highcharts_data(page)
            if eod_data: push_to_firebase(eod_data, "EOD")
    except:
        pass
    return "Contract"

def select_best_secondary_series(page, current_main_series_code="OGJ6"):
    options = page.locator(".link-selector ul.nav li a").all()
    candidates = []
    now = datetime.now()
    for opt in options:
        try:
            full_text = opt.inner_text().strip()
            lines = full_text.split('\n')
            if len(lines) >= 2:
                code, date_str = lines[0].strip(), lines[1].strip()
                exp_date = datetime.strptime(date_str, "%d %b %Y")
                if is_series_active(exp_date, now) and code != current_main_series_code:
                    candidates.append({'code': code, 'date': exp_date, 'element': opt})
        except: continue
    if candidates:
        candidates.sort(key=lambda x: x['date'])
        candidates[0]['element'].click(force=True)
        return candidates[0]['code']
    return None

def run_bot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()
        page.goto("https://cmegroup-tools.quikstrike.net//User/QuikStrikeView.aspx?pid=40&pf=6&viewitemid=IntegratedV2VExpectedRange&insid=209341331&qsid=baf9e29e-d0c4-4680-bac6-5b07eabe76b5")
        page.wait_for_load_state("domcontentloaded")
        time.sleep(8) 
        main_contract = process_current_series(page, "Main_Monthly")
        try:
            page.locator("text='Expiration:' >> visible=true").first.click(force=True)
            time.sleep(2)
            selected_code = select_best_secondary_series(page, main_contract)
            if selected_code:
                time.sleep(8)
                try:
                    intraday_btn = page.locator("text='Intraday Volume' >> visible=true")
                    if intraday_btn.count() > 0:
                        intraday_btn.first.click(force=True)
                        time.sleep(5)
                except: pass
                process_current_series(page, f"Secondary_{selected_code}")
        except: pass
        browser.close()

if __name__ == "__main__":
    run_bot()
