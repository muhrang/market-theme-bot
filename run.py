import os, nest_asyncio, asyncio, cv2, numpy as np, re, pandas as pd, pytesseract, requests
from playwright.async_api import async_playwright
import easyocr
from jamo import h2j
import Levenshtein
from openai import OpenAI
import FinanceDataReader as fdr

nest_asyncio.apply()

# --- 🔥 시크릿에서 불러오기 (하드코딩 금지) ---
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID")

if not OPENAI_KEY: raise Exception("❌ OPENAI_API_KEY 없음 (Secrets 확인)")
if not TELEGRAM_BOT: raise Exception("❌ TELEGRAM_BOT_TOKEN 없음 (Secrets 확인)")
if not TELEGRAM_CHAT: raise Exception("❌ TELEGRAM_CHAT_ID 없음 (Secrets 확인)")

client = OpenAI(api_key=OPENAI_KEY)

# ---- 1) 화면 캡처 ----
async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage","--disable-gpu"]
        )
        page = await browser.new_page(viewport={"width":1920,"height":3000,"deviceScaleFactor":2})
        await page.goto("https://wts.ls-sec.co.kr/#0021", timeout=20000000, wait_until="networkidle")
        await page.wait_for_timeout(1000)
        await page.mouse.click(360, 165)
        await page.wait_for_timeout(1000)
        await page.screenshot(path="after.png", full_page=True)
        await browser.close()

asyncio.run(capture())

# ---- 2) OCR ----
reader = easyocr.Reader(['ko','en'])
img = cv2.imread("after.png")
Y1, Y2 = 230, 2800
X1, X2 = 100, 430
Xr1, Xr2 = 1250, 1340
Xv1, Xv2 = 1700, 1820
table = img[Y1:Y2]

g = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
edges = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
edges = cv2.convertScaleAbs(edges)
proj = edges.sum(axis=1)
smooth = np.convolve(proj, np.ones(13)/13, mode='same')
thr = np.percentile(smooth, 75)
cands = np.where(smooth>thr)[0]
rows=[]; buf=[cands[0]]
for v in cands[1:]:
    if v-buf[-1]<=26: buf.append(v)
    else: rows.append(int(np.mean(buf))); buf=[v]
rows.append(int(np.mean(buf)))

def fix_rate(t):
    raw = re.sub(r"[^0-9.+-]", "", str(t))
    if raw=="": return None
    sign = "-" if raw.startswith("-") else "+"
    raw = raw.replace("+","").replace("-","")
    try: val=float(raw)
    except: val=float(raw[:-2]+"."+raw[-2:])
    while abs(val)>30: val/=10
    return f"{sign}{abs(val):.2f}%"

def best_number(lst):
    nums=[''.join(ch for ch in str(x) if ch.isdigit()) for x in lst]
    nums=[n for n in nums if len(n)>=3]
    return max(nums, key=len) if nums else None

records=[]
for cy in rows:
    y1=max(0,cy-26); y2=min(table.shape[0],y1+52)
    line=table[y1:y2]
    name_raw = reader.readtext(line[:,X1:X2], detail=0)
    name=name_raw[0].strip() if name_raw else None
    rate=fix_rate(pytesseract.image_to_string(cv2.cvtColor(line[:,Xr1:Xr2], cv2.COLOR_BGR2GRAY)))
    val=best_number(reader.readtext(line[:,Xv1:Xv2], detail=0))
    if name and val: records.append([name,rate,val])

df=pd.DataFrame(records, columns=["종목명","등락률","거래대금"])

# ---- 3) 종목명 보정 / +5% 필터 ----
names = fdr.StockListing("KRX")["Name"].tolist()
def correct(n):
    score=[(s,Levenshtein.distance(h2j(n),h2j(s))) for s in names]
    score.sort(key=lambda x:x[1])
    return score[0][0] if score[0][1]<=3 else n

df["종목명"]=df["종목명"].apply(lambda x:x if x in names else correct(x))
df=df[~df["종목명"].str.contains("레버|인버|ETF|ETN|선물|KODEX|TIGER")]
df["거래대금"]=df["거래대금"].astype(int)
df["등락률_float"]=df["등락률"].str.extract(r'([+-]?\d+\.?\d*)').astype(float)
df=df.sort_values("거래대금",ascending=False).head(30)
df=df[df["등락률_float"]>=5].copy()

# ---- 4) GPT 테마 묶기 ----
rows=[f"{r['종목명']} | {r['등락률_float']:.2f}% | {int(r['거래대금'])}" for _,r in df.iterrows()]
prompt="\n".join(rows)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":f"테마별로 묶어서 설명해줘:\n{prompt}"}],
    max_tokens=500
)
out=resp.choices[0].message.content

# ---- 5) 텔레그램 전송 ----
msg=f"📈 +5% 강세 종목 테마 분석\n\n{out}"
requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage",
    params={"chat_id":TELEGRAM_CHAT, "text":msg})

print("✅ 텔레그램 전송 완료")

from telegram.ext import Updater, CommandHandler
import threading, time

running = False   # 실행 여부 제어 변수

def job_loop():
    global running
    while running:
        print("📡 데이터 수집 & 분석 실행중...")
        try:
            asyncio.run(capture())     # 기존 캡처
            # 아래 기존 분석 + GPT + 텔레그램 보내는 부분 그대로
        except Exception as e:
            print("❌ 오류:", e)
        time.sleep(30)  # 30초마다 반복 (원하면 수정 가능)

def start_cmd(update, context):
    global running
    if running:
        update.message.reply_text("이미 실행중 ✅")
        return
    running = True
    threading.Thread(target=job_loop, daemon=True).start()
    update.message.reply_text("🚀 자동모드 시작!")

def stop_cmd(update, context):
    global running
    running = False
    update.message.reply_text("⛔ 자동모드 정지!")

def status_cmd(update, context):
    update.message.reply_text("상태: " + ("실행중 ✅" if running else "정지 ⏸"))

def enable_remote_control():
    updater = Updater(TELEGRAM_BOT, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_cmd))
    dp.add_handler(CommandHandler("stop", stop_cmd))
    dp.add_handler(CommandHandler("status", status_cmd))
    updater.start_polling()
    print("📱 Telegram Remote Control Ready")
    updater.idle()

if __name__ == "__main__":
    enable_remote_control()   # 🔥 항상 명령 대기 상태

