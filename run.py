import os, requests, pandas as pd, re, numpy as np, cv2, pytesseract, asyncio, nest_asyncio
from playwright.async_api import async_playwright
import easyocr, FinanceDataReader as fdr
from jamo import h2j
import Levenshtein
from openai import OpenAI

# === 기본 설정 ===
TELEGRAM_CHAT_ID = "6647068566"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # GitHub Secrets 로 설정할 예정
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")              # GitHub Secrets 로 설정할 예정
client = OpenAI(api_key=OPENAI_KEY)

nest_asyncio.apply()

async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width":1920,"height":2700,"deviceScaleFactor":2})
        await page.goto("https://wts.ls-sec.co.kr/#0021", timeout=20000000, wait_until="networkidle")
        await page.wait_for_timeout(900)
        await page.mouse.click(360,165)
        await page.wait_for_timeout(900)
        await page.screenshot(path="screen.png", full_page=True)
        await browser.close()

asyncio.run(capture())

reader = easyocr.Reader(["ko","en"])
img = cv2.imread("screen.png")

Y1,Y2 = 230,2800
X_NAME_L,X_NAME_R = 100,430
X_RATE_L,X_RATE_R = 1250,1340
X_VALUE_L,X_VALUE_R = 1700,1820

table = img[Y1:Y2]
g = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
edges = cv2.Sobel(g, cv2.CV_16S, 0,1,ksize=3)
edges = cv2.convertScaleAbs(edges)
proj = edges.sum(axis=1)
smooth = np.convolve(proj,np.ones(13)/13,mode="same")
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
    raw=raw.replace("+","").replace("-","")
    try: v=float(raw) if "." in raw else float(raw[:-2]+"."+raw[-2:])
    except: return None
    while abs(v)>30: v/=10
    return f"{sign}{abs(v):.2f}%"

def best(lst):
    nums=[''.join(ch for ch in str(x) if ch.isdigit()) for x in lst]
    nums=[n for n in nums if len(n)>=3]
    return max(nums,key=len) if nums else None

data=[]
for cy in rows:
    y1=max(0,cy-26); y2=min(table.shape[0],y1+52)
    line=table[y1:y2]
    name_r=reader.readtext(line[:,X_NAME_L:X_NAME_R],detail=0)
    name=name_r[0].strip() if name_r else None
    rate=fix_rate(pytesseract.image_to_string(cv2.cvtColor(line[:,X_RATE_L:X_RATE_R],cv2.COLOR_BGR2GRAY)))
    val=best(reader.readtext(line[:,X_VALUE_L:X_VALUE_R],detail=0))
    if name and val: data.append([name,rate,val])

df=pd.DataFrame(data,columns=["종목명","등락률","거래대금"])

krx=fdr.StockListing("KRX")
names=krx["Name"].tolist()

def fix_name(n):
    s=[(t,Levenshtein.distance(h2j(n),h2j(t))) for t in names]
    s.sort(key=lambda x:x[1])
    return s[0][0] if s[0][1]<=3 else n

df["종목명"]=df["종목명"].apply(lambda x: x if x in names else fix_name(x))
df=df[~df["종목명"].str.contains("레버|인버|ETF|ETN|KODEX|TIGER")]
df["거래대금"]=df["거래대금"].astype(int)
df["등락률_float"]=df["등락률"].str.extract(r"([+-]?\d+\.?\d*)").astype(float)

df=df.sort_values("거래대금",ascending=False).head(30)
df=df[df["등락률_float"]>=5].copy()

txt="\n".join([f"{r['종목명']} | {r['등락률']} | {r['거래대금']}" for _,r in df.iterrows()])

prompt=f"""
다음 종목들을 산업/정책/수급 흐름 기반으로 테마 그룹화하고 대장주를 1개 골라.
형식:
테마:
 - 설명
 - 종목들: A, B, C

종목:
{txt}
"""

res=client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
out=res.choices[0].message.content

msg=f"📈 오늘 시장 테마 분석 결과\n\n{out}"

requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
             params={"chat_id":TELEGRAM_CHAT_ID,"text":msg})
print("완료")
