import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
import cv2
import numpy as np
import re
import pandas as pd
import pytesseract
import requests
from playwright.async_api import async_playwright
import easyocr
from jamo import h2j
import Levenshtein
from openai import OpenAI

# ===============================
# ✅ 환경변수 (GitHub Secrets 사용)
# ===============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not OPENAI_API_KEY:
    raise SystemExit("❌ OPENAI_API_KEY 없음 (GitHub Secrets에 추가 필요)")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# ✅ KRX 종목 리스트 (FDR 제거)
# ===============================
krx_url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
krx = pd.read_html(krx_url, header=0)[0]
krx['Code'] = krx['종목코드'].astype(str).str.zfill(6)
names = krx["회사명"].tolist()
code_map = dict(zip(krx["회사명"], krx["Code"]))

# ===============================
# ✅ LS WTS 화면 캡처
# ===============================
async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage","--disable-gpu"]
        )
        page = await browser.new_page(viewport={"width":1920,"height":3000,"deviceScaleFactor":2})
        await page.goto("https://wts.ls-sec.co.kr/#0021", wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.mouse.click(360, 165)
        await page.wait_for_timeout(800)
        await page.screenshot(path="after_click.png", full_page=True)
        await browser.close()

asyncio.run(capture())

# ===============================
# ✅ OCR 데이터 추출
# ===============================
reader = easyocr.Reader(['ko','en'])
tess_cfg = '--psm 7 -c tessedit_char_whitelist=+-0123456789.%'

img = cv2.imread("after_click.png")
Y1, Y2 = 230, 2800
X_NAME_L, X_NAME_R = 100, 430
X_RATE_L, X_RATE_R = 1250, 1340
X_VALUE_L, X_VALUE_R = 1700, 1820
table = img[Y1:Y2]

g = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
edges = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
edges = cv2.convertScaleAbs(edges)
proj = edges.sum(axis=1)
smooth = np.convolve(proj, np.ones(13)/13, mode='same')
thr = np.percentile(smooth, 75)
cands = np.where(smooth > thr)[0]

rows=[]
buf=[cands[0]]
for v in cands[1:]:
    if v-buf[-1] <= 26: buf.append(v)
    else: rows.append(int(np.mean(buf))); buf=[v]
rows.append(int(np.mean(buf)))

def fix_rate(txt):
    raw = re.sub(r"[^0-9.+-]", "", str(txt))
    if raw == "": return None
    sign = "-" if raw.startswith("-") else "+"
    raw = raw.replace("+","").replace("-","")
    try: val = float(raw) if "." in raw else float(raw[:-2] + "." + raw[-2:])
    except: return None
    while abs(val)>30: val/=10
    return f"{sign}{abs(val):.2f}%"

def best_number(lst):
    nums = [''.join(ch for ch in str(x) if ch.isdigit()) for x in lst]
    nums = [n for n in nums if len(n)>=3]
    return max(nums, key=len) if nums else None

records=[]
for cy in rows:
    y1=max(0,cy-26); y2=min(table.shape[0], y1+52)
    line = table[y1:y2]
    name_raw = reader.readtext(line[:,X_NAME_L:X_NAME_R], detail=0)
    name = name_raw[0].strip() if name_raw else None
    rate = fix_rate(pytesseract.image_to_string(cv2.cvtColor(line[:,X_RATE_L:X_RATE_R], cv2.COLOR_BGR2GRAY), config=tess_cfg))
    vol = best_number(reader.readtext(line[:,X_VALUE_L:X_VALUE_R], detail=0))
    if name and vol: records.append([name, rate, vol])

df = pd.DataFrame(records, columns=["종목명","등락률","거래대금"])

def correct_name(name):
    score = [(s, Levenshtein.distance(h2j(name), h2j(s))) for s in names]
    score.sort(key=lambda x:x[1])
    return score[0][0] if score[0][1] <= 3 else name

df["종목명"] = df["종목명"].apply(lambda x: x if x in names else correct_name(x))
df = df[~df["종목명"].str.contains("레버|인버|ETF|ETN|선물|KODEX|TIGER")]
df["등락률_float"] = df["등락률"].str.extract(r'([+-]?\d+\.?\d*)').astype(float)
df["거래대금"] = df["거래대금"].astype(int)

df_top30 = df.sort_values("거래대금", ascending=False).head(30)
df_filtered = df_top30[df_top30["등락률_float"] >= 5].copy()

# ===============================
# ✅ GPT 테마 분석
# ===============================
rows = [f"{r['종목명']} | {r['등락률_float']:.2f}% | {int(r['거래대금'])}" for _, r in df_filtered.iterrows()]
prompt_items = "\n".join(rows)

prompt = f"""
다음 종목들을 테마별로 묶고, 오늘 시장 주도 테마와 대장주를 알려줘.

형식:
테마:
 - 설명
 - 종목들: A, B, C

데이터:
{prompt_items}
"""

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":prompt}],
    max_tokens=800
)
gpt_output = resp.choices[0].message.content

# ===============================
# ✅ 메시지 → 텔레그램 전송
# ===============================
def send(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                     params={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

send("📈 오늘 테마 분석 결과\n\n" + gpt_output)
print("✅ 완료")
