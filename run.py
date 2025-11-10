import os, asyncio, nest_asyncio, cv2, numpy as np, re, pandas as pd, pytesseract, requests
from playwright.async_api import async_playwright
import easyocr
from jamo import h2j
import Levenshtein
from openai import OpenAI

nest_asyncio.apply()

# ===== Secrets =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not OPENAI_API_KEY:
    raise Exception("❌ OPENAI_API_KEY 없음 (GitHub Secrets 확인)")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise Exception("❌ Telegram 설정 없음")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== KRX 종목명 리스트 (FDR 대체) =====
krx_url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
krx = pd.read_excel(krx_url)
names = krx["회사명"].tolist()

# ===== 화면 캡처 =====
async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width":1920,"height":3000,"deviceScaleFactor":2})
        await page.goto("https://wts.ls-sec.co.kr/#0021", timeout=20000000, wait_until="networkidle")
        await page.wait_for_timeout(1000)
        await page.mouse.click(360,165)
        await page.wait_for_timeout(800)
        await page.screenshot(path="after.png", full_page=True)
        await browser.close()

asyncio.run(capture())

# ===== OCR =====
reader = easyocr.Reader(['ko','en'])
img = cv2.imread("after.png")

Y1,Y2 = 230,2800
X_NAME=(100,430); X_RATE=(1250,1340); X_VALUE=(1700,1820)

table = img[Y1:Y2]
g = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
edges = cv2.Sobel(g, cv2.CV_16S,0,1,ksize=3)
edges = cv2.convertScaleAbs(edges)
proj = edges.sum(axis=1)
smooth = np.convolve(proj, np.ones(13)/13, mode='same')
thr = np.percentile(smooth,75)
cands = np.where(smooth>thr)[0]

rows=[]; buf=[cands[0]]
for v in cands[1:]:
    if v-buf[-1]<=26: buf.append(v)
    else: rows.append(int(np.mean(buf))); buf=[v]
rows.append(int(np.mean(buf)))

def fix_rate(t):
    raw = re.sub(r"[^0-9.+-]","",str(t))
    if raw=="": return None
    sign = "-" if raw.startswith("-") else "+"
    raw = raw.replace("+","").replace("-","")
    try: val = float(raw) if "." in raw else float(raw[:-2]+"."+raw[-2:])
    except: return None
    while abs(val)>30: val/=10
    return f"{sign}{abs(val):.2f}%"

def best_num(l):
    nums=[''.join(ch for ch in str(x) if ch.isdigit()) for x in l]
    nums=[n for n in nums if len(n)>=3]
    return max(nums,key=len) if nums else None

records=[]
for cy in rows:
    line = table[max(0,cy-26):max(0,cy-26)+52]
    name_raw = reader.readtext(line[:,X_NAME[0]:X_NAME[1]], detail=0)
    rate = fix_rate(pytesseract.image_to_string(cv2.cvtColor(line[:,X_RATE[0]:X_RATE[1]],cv2.COLOR_BGR2GRAY)))
    val = best_num(reader.readtext(line[:,X_VALUE[0]:X_VALUE[1]], detail=0))
    if name_raw and val: records.append([name_raw[0],rate,val])

df = pd.DataFrame(records,columns=["종목명","등락률","거래대금"])

# 종목 교정
def correct(s):
    score=[(n,Levenshtein.distance(h2j(s),h2j(n))) for n in names]
    score.sort(key=lambda x:x[1])
    return score[0][0] if score[0][1]<=3 else s

df["종목명"]=df["종목명"].apply(lambda x: x if x in names else correct(x))
df=df[~df["종목명"].str.contains("레버|인버|ETF|ETN|선물|KODEX|TIGER")]
df["거래대금"]=df["거래대금"].astype(int)
df["등락률_float"]=df["등락률"].str.extract(r'([+-]?\d+\.?\d*)').astype(float)

df=df.sort_values("거래대금",ascending=False).head(30)
df=df[df["등락률_float"]>=5].copy()

# GPT 테마
rows=[f"{r['종목명']} | {r['등락률_float']}% | {r['거래대금']}" for _,r in df.iterrows()]
prompt="데이터:\n"+("\n".join(rows))+"\n\n이를 3~6개 테마로 묶어줘. 형식:\n테마:\n - 설명\n - 종목들:"

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":prompt}],
    max_tokens=800
)
txt = resp.choices[0].message.content

theme={}
for block in txt.split("\n\n"):
    if ":" not in block: continue
    title = block.splitlines()[0].strip().rstrip(":")
    for ln in block.splitlines():
        if "종목" in ln:
            part = ln.split(":",1)[1]
            for n in [x.strip() for x in part.split(",")]:
                for real in df["종목명"]:
                    if n==real or n in real or real in n:
                        theme[real]=title

df["테마"]=df["종목명"].map(theme).fillna("기타")
top = df["테마"].value_counts().idxmax()
dai = df[df["테마"]==top].sort_values("등락률_float",ascending=False).iloc[0]

msg = f"🔥 주도 테마: {top}\n⭐ 대장주: {dai['종목명']} ({dai['등락률']})\n거래대금: {dai['거래대금']:,}"

requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
             params={"chat_id":TELEGRAM_CHAT_ID,"text":msg})
