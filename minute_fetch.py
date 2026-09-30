"""단타 연습용 1분봉 수집 (GitHub Actions에서 실행)

네이버 분봉 API는 최근 약 8거래일치만 준다. 그래서 매 평일 돌려서 minute/YYYYMMDD.json 으로 쌓아 둔다.
- 대상: 그날 거래대금 상위 TOP_N 종목(ETF/ETN 제외). 단타 대상이 몰리는 종목 위주.
- 정규장(09:00~15:29) 390분만 저장. 거래가 없어 빈 분은 직전 종가로 채움(거래량 0).
- 파일 형식: 종목마다 cl(종가), o/h/l(종가와의 차이), v(분당 거래량 주식수) 배열 길이 390.
- 같은 날 파일이 이미 있으면 종목 단위로 새 값으로 덮어쓰고 나머지는 유지. 오래된 파일은 KEEP_DAYS일만 남김.
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import scan_ma as S

TOP_N = 150
KEEP_DAYS = 40
WORKERS = 6
OUT = os.path.join(S.BASE, "minute")
SLOTS = 390   # 09:00 ~ 15:29


def fetch_minutes(code, start, end):
    r = S.http_get(f"https://api.stock.naver.com/chart/domestic/item/{code}/minute",
                   params={"startDateTime": start, "endDateTime": end}, timeout=30)
    return r.json()


def by_day(rows):
    """rows -> {날짜: {slot: (o,h,l,c,acc)}}"""
    days = defaultdict(dict)
    for r in rows:
        t = str(r["localDateTime"])
        hm = int(t[8:10]) * 60 + int(t[10:12])
        slot = hm - 540
        if 0 <= slot < SLOTS:
            days[t[:8]][slot] = (r["openPrice"], r["highPrice"], r["lowPrice"], r["currentPrice"], r.get("accumulatedTradingVolume") or 0)
    return days


def pack(day_bars, prev_close):
    """빈 분 채우기 + 정수 배열로"""
    first = min(day_bars)
    o0 = day_bars[first][0]
    pc = int(round(prev_close if prev_close else o0))
    cl, dO, dH, dL, vol = [], [], [], [], []
    last_c, last_acc = pc, 0
    for k in range(SLOTS):
        b = day_bars.get(k)
        if b is None:
            c = int(round(last_c))
            cl.append(c); dO.append(0); dH.append(0); dL.append(0); vol.append(0)
            continue
        o, h, l, c, acc = b
        c = int(round(c)); o = int(round(o)); h = int(round(h)); l = int(round(l))
        h = max(h, o, c); l = min(l, o, c)
        cl.append(c); dO.append(o - c); dH.append(h - c); dL.append(l - c)
        vol.append(max(0, int(acc - last_acc)))
        last_c, last_acc = c, acc
    return pc, {"cl": cl, "o": dO, "h": dH, "l": dL, "v": vol}


def main():
    t0 = time.time()
    S.LOG_PATH = os.path.join(S.BASE, "minute_log.txt")
    names = S.get_ticker_list()
    if len(names) < 1000:
        S.log("종목 목록 수집 실패"); sys.exit(1)
    meta = S.load_meta()
    ok = meta[(meta["구분"] == "stock") & (meta["거래상태"] == "tradable")]
    top = ok.sort_values("거래대금", ascending=False).head(TOP_N)
    codes = list(top.index)
    S.log(f"분봉 대상 {len(codes)}종목 (거래대금 상위)")
    now = datetime.now()
    start = (now - timedelta(days=16)).strftime("%Y%m%d") + "0900"
    end = now.strftime("%Y%m%d") + "1600"

    per = {}
    fails = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_minutes, c, start, end): c for c in codes}
        for f in as_completed(futs):
            c = futs[f]
            try:
                per[c] = by_day(f.result())
            except Exception as e:
                fails += 1
                if fails <= 3:
                    S.log("실패", c, repr(e)[:120])
    S.log(f"수신 {len(per)}종목, 실패 {fails}, {time.time() - t0:.0f}초")
    if len(per) < len(codes) * 0.5:
        S.log("실패가 너무 많아 저장하지 않음"); sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    all_days = sorted({d for m in per.values() for d in m})
    for d in all_days:
        path = os.path.join(OUT, f"{d}.json")
        cur = {}
        if os.path.exists(path):
            try:
                cur = {s["c"]: s for s in json.load(open(path, encoding="utf-8"))["stocks"]}
            except Exception:
                cur = {}
        for c, days in per.items():
            if d not in days or len(days[d]) < 60:
                continue
            prevs = [x for x in sorted(days) if x < d]
            prev_close = None
            if prevs:
                pb = days[prevs[-1]]
                prev_close = pb[max(pb)][3]
            pc, arr = pack(days[d], prev_close)
            cur[c] = {"c": c, "n": str(meta.loc[c, "종목명"]), "m": str(meta.loc[c, "시장"]), "pc": pc, **arr}
        if cur:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"date": d, "stocks": list(cur.values())}, f, ensure_ascii=False, separators=(",", ":"))
    files = sorted(x[:-5] for x in os.listdir(OUT) if x.endswith(".json") and x != "index.json")
    for old in files[:-KEEP_DAYS]:
        os.remove(os.path.join(OUT, old + ".json"))
    files = files[-KEEP_DAYS:]
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump({"days": files}, f)
    S.log(f"minute 저장 완료: {len(files)}일 ({files[0] if files else '-'} ~ {files[-1] if files else '-'})")


if __name__ == "__main__":
    main()
