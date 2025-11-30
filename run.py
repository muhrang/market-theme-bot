import os, nest_asyncio, asyncio, cv2, numpy as np, re, pandas as pd, pytesseract, requests
from playwright.async_api import async_playwright
import easyocr
from jamo import h2j
import Levenshtein
from openai import OpenAI
import FinanceDataReader as fdr
import time

nest_asyncio.apply()

# --- 🔥 시크릿 키 ---
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID")

client = OpenAI(api_key=OPENAI_KEY)

reader = easyocr.Reader(['ko','en'])

# ✅ 브라우저/페이지 전역 유지
browser = None
page = None

# ---- 1) 최초 1회만 브라우저 오픈 ----
async def init_browser():
    global browser, page
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage","--disable-gpu"]
    )
    page = await browser.new_page(viewport={"width":1920,"height":3000,"deviceScaleFactor":2})

    await page.goto("https://wts.ls-sec.co.kr/#0021", timeout=20000000)
    await page.wait_for_timeout(3000)
    await page.mouse.click(360, 165)
    await page.wait_for_timeout(2000)

    print("✅ 브라우저 최초 세팅 완료")

# ---- 2) 이후엔 캡처만 반복 ----
async def fast_capture():
    global page
    await page.reload()
    await page.wait_for_timeout(2000)
    await page.screenshot(path="after.png", full_page=True)

# ---- OCR 보정 로직 유지 ----
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

# ---- 분석 + 텔레그램 전송 ----
def analyze_and_send():
    img = cv2.imread("after.png")
    table = img[230:2800]

    g = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    proj = edges.sum(axis=1)
    smooth = np.convolve(proj, np.ones(13)/13, mode='same')

    rows = np.where(smooth > np.percentile(smooth, 75))[0]

    records=[]
    for cy in rows[::8]:
        line=table[cy:cy+52]
        name_raw = reader.readtext(line[:,100:430], detail=0)
        name=name_raw[0].strip() if name_raw else None
        rate=fix_rate(pytesseract.image_to_string(line[:,1250:1340]))
        val=best_number(reader.readtext(line[:,1700:1820], detail=0))
        if name and val: records.append([name,rate,val])

    df=pd.DataFrame(records, columns=["종목명","등락률","거래대금"])

    names = fdr.StockListing("KRX")["Name"].tolist()

    def correct(n):
        score=[(s,Levenshtein.distance(h2j(n),h2j(s))) for s in names]
        score.sort(key=lambda x:x[1])
        return score[0][0]

    df["종목명"]=df["종목명"].apply(lambda x: correct(x) if x not in names else x)

    df=df[~df["종목명"].str.contains("레버|인버|ETF|ETN|선물|KODEX|TIGER")]
    df["거래대금"]=df["거래대금"].astype(int)
    df["등락률_float"]=df["등락률"].str.extract(r'([+-]?\d+\.?\d*)').astype(float)

    df=df.sort_values("거래대금",ascending=False).head(30)
    df=df[df["등락률_float"]>=5]

    if df.empty:
        print("⏳ +5% 강세 종목 없음")
        return

    rows_text=[f"{r['종목명']} | {r['등락률_float']:.2f}% | {int(r['거래대금'])}" for _,r in df.iterrows()]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"테마별로 묶어서 설명해줘:\n" + "\n".join(rows_text)}],
        max_tokens=500
    )

    out=resp.choices[0].message.content

    requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage",
        params={"chat_id":TELEGRAM_CHAT, "text":f"📈 +5% 강세 종목\n\n{out}"}
    )

    print("✅ 텔레그램 전송 완료")

# ---- ✅ 30초 루프 ----
async def main_loop():
    await init_browser()
    start = time.time()

    while True:
        if time.time() - start > 6*60*60:
            print("⏹ 6시간 종료")
            break

        try:
            await fast_capture()
            analyze_and_send()
        except Exception as e:
            print("❌ 오류:", e)

        await asyncio.sleep(30)

asyncio.run(main_loop())
