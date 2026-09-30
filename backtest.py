"""이평선 눌림 후 반등 전략 백테스트 (GitHub Actions에서 실행, 네이버 시세 사용)

규칙 (사용자가 고른 방식):
- 진입: 이평선(60/112/224) 눌림 후 반등 -> 다음날 시가에 매수
- 청산: 정해진 % 수익(목표가) / -5% 손절 / (안전장치) 60거래일 지나면 종가 청산
- 분산: 최대 3종목 집중 (포트폴리오 시뮬레이션)

주의: 현재 상장 종목만, 오늘 기준 시총/거래대금 필터로 고른 종목이라 실제보다 좋게 나오는 쪽으로 편향됨.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import scan_ma as S

MAS = [60, 112, 224]
TARGETS = [5, 8, 10, 15, 20]     # 목표 수익 %
STOP = 5.0                       # 손절 %
MAXHOLD = 60                     # 최대 보유 거래일
COST = 0.30                      # 왕복 비용(수수료+거래세+슬리피지 가정) %
SLOTS = 3                        # 동시 보유 종목 수
BASE_SAMPLES = 40                # 비교군: 종목당 무작위 진입 횟수
MIN_VALUE = 1e9                  # 신호일 거래대금 하한
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt_result.md")


def load():
    S.COUNT = 1700   # 약 6.5년
    S.get_ticker_list()
    meta = S.load_meta()
    want = meta.index[
        meta["구분"].eq("stock") & meta["거래상태"].eq("tradable")
        & (meta["시총"].fillna(0) >= S.MIN_MARKET_CAP) & (meta["거래대금"].fillna(0) >= S.MIN_TRADING_VALUE)
    ].tolist()
    S.log(f"백테스트 대상 {len(want)}종목 시세 받는 중")
    rows, fails = S.download_all(want)
    if not rows:
        S.log("시세 없음")
        sys.exit(1)
    cache = pd.DataFrame(rows, columns=["날짜", "티커", "시가", "고가", "저가", "종가", "거래대금"])
    px = S.get_px(cache)
    oh = S.get_ohlc(cache, px)
    val = cache.pivot(index="날짜", columns="티커", values="거래대금").reindex(index=px.index, columns=px.columns)
    S.log(f"수신 {px.shape[1]}종목, {px.shape[0]}거래일 ({px.index[0]}~{px.index[-1]}), 실패 {len(fails)}")
    return px, oh, val


def sim(o, h, l, c, i, target, stop=STOP, maxh=MAXHOLD):
    """i번째 봉 시가에 매수. (수익률 소수, 보유일, 청산 인덱스) 또는 None(끝까지 안 끝난 거래)."""
    e = o[i]
    if not e > 0:
        return None
    tp, sl = e * (1 + target / 100), e * (1 - stop / 100)
    n = len(c)
    last = i + maxh - 1
    for j in range(i, min(n, last + 1)):
        if j > i:
            if o[j] <= sl:
                return o[j] / e - 1, j - i, j       # 갭하락: 시가에 손절
            if o[j] >= tp:
                return o[j] / e - 1, j - i, j       # 갭상승: 시가에 익절
        if l[j] <= sl:
            return sl / e - 1, j - i, j             # 같은 날 손절/익절 동시면 손절 먼저(보수적)
        if h[j] >= tp:
            return tp / e - 1, j - i, j
    if last < n:
        return c[last] / e - 1, last - i, last      # 시간 청산
    return None


def stock_signals(o, h, l, c, v, N):
    ma = pd.Series(c).rolling(N).mean()
    ratio = pd.Series(c / ma)
    trend = ma > ma.shift(20)                                    # 이평선이 우상향
    touch = pd.Series(l / ma).rolling(5).min() <= 1.03           # 최근 5일 안에 이평선 +3% 이내까지 눌림
    deep = ratio.shift(6).rolling(25).max() >= 1.10              # 그 전에는 이평선보다 10% 이상 위에 있었음(진짜 눌림)
    hold = c > ma                                                # 종가는 이평선 위 유지
    prev = np.r_[np.nan, c[:-1]]
    bull = (c > o) & (c > prev)                                  # 양봉 + 전일 종가 위 (반등)
    liq = v >= MIN_VALUE
    base = (trend & hold & liq).values
    sig = (trend & touch & deep & hold & bull & liq).values
    return sig, base


def run(px, oh, val, log=print):
    dates = np.array(px.index)
    mid = dates[len(dates) // 2]
    rng = np.random.default_rng(7)
    per = {}   # (N, X) -> dict(trades=[...], base=[...])
    for N in MAS:
        for X in TARGETS:
            per[(N, X)] = {"tr": [], "bs": []}
    for code in px.columns:
        c0 = px[code].values
        ok = ~np.isnan(c0)
        if ok.sum() < 400:
            continue
        d = dates[ok]
        o, h, l, c = (oh["o"][code].values[ok], oh["h"][code].values[ok], oh["l"][code].values[ok], c0[ok])
        v = np.nan_to_num(val[code].values[ok])
        n = len(c)
        for N in MAS:
            sig, base = stock_signals(o, h, l, c, v, N)
            sig_idx = np.where(sig[:-1])[0] + 1          # 신호 다음날 시가에 진입
            base_idx = np.where(base[:-1])[0] + 1
            if len(base_idx) > BASE_SAMPLES:
                base_idx = rng.choice(base_idx, BASE_SAMPLES, replace=False)
            for X in TARGETS:
                rec = per[(N, X)]
                busy = -1
                for i in sig_idx:
                    if i <= busy:
                        continue
                    r = sim(o, h, l, c, i, X)
                    if r is None:
                        continue
                    ret, days, j = r
                    busy = j
                    rec["tr"].append((d[i], d[j], ret * 100 - COST, days, v[i - 1], code))
                for i in base_idx:
                    r = sim(o, h, l, c, i, X)
                    if r is not None:
                        rec["bs"].append((d[i], r[0] * 100 - COST, r[1]))
    return per, mid


def stats(rets, days):
    r = np.array(rets, dtype=float)
    if len(r) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan, hold=np.nan)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, avg=r.mean(), pf=(pos / neg if neg > 0 else np.inf), hold=np.mean(days))


def portfolio(trades, slots=SLOTS):
    """trades: (entry, exit, ret%, days, value, code). 최대 slots종목, 신호가 몰리면 거래대금 큰 순으로 채움."""
    tr = sorted(trades, key=lambda t: (t[0], -t[4]))
    cash, open_, curve = 1.0, [], []
    for e, x, ret, days, val, code in tr:
        rel = sorted([p for p in open_ if p[0] < e], key=lambda p: p[0])
        open_ = [p for p in open_ if p[0] >= e]
        for p in rel:
            cash += p[1] * (1 + p[2] / 100)
        if rel:
            curve.append((rel[-1][0], cash + sum(q[1] for q in open_)))
        if len(open_) < slots:
            total = cash + sum(p[1] for p in open_)
            alloc = min(cash, total / slots)
            if alloc <= 0:
                continue
            cash -= alloc
            open_.append((x, alloc, ret))
    for p in open_:
        cash += p[1] * (1 + p[2] / 100)
    if open_:
        curve.append((max(p[0] for p in open_), cash))
    if not curve:
        return dict(total=np.nan, cagr=np.nan, mdd=np.nan)
    eq = np.array([c for _, c in curve])
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = ((eq / peak) - 1).min() * 100
    d0, d1 = pd.Timestamp(tr[0][0]), pd.Timestamp(tr[-1][1])
    yrs = max((d1 - d0).days / 365.25, 0.1)
    return dict(total=(eq[-1] - 1) * 100, cagr=(eq[-1] ** (1 / yrs) - 1) * 100, mdd=mdd)


def fmt(x, p=1):
    return "-" if x is None or (isinstance(x, float) and (np.isnan(x))) else ("inf" if x == np.inf else f"{x:.{p}f}")


def report(per, mid, px):
    L = []
    L.append("# 이평선 눌림 후 반등 백테스트\n")
    L.append(f"- 기간: {px.index[0]} ~ {px.index[-1]}, 종목 {px.shape[1]}개 (오늘 기준 시총 500억+, 거래대금 10억+, ETF/ETN 제외)")
    L.append(f"- 진입: 이평선 눌림 후 반등(양봉) 다음날 시가 / 청산: 목표수익 or -{STOP:g}% 손절 or {MAXHOLD}거래일 종가")
    L.append(f"- 비용 가정: 왕복 {COST}% 차감. 같은 날 손절·익절이 겹치면 손절로 계산. 종목당 동시에 1포지션.")
    L.append("- 주의: 현재 상장 종목만 대상(상장폐지 종목 제외)이고 필터도 오늘 기준이라 **실제보다 좋게 나오는 쪽**으로 편향됨.\n")
    for N in MAS:
        L.append(f"## {N}일선 눌림 후 반등\n")
        L.append("| 목표 | 거래수 | 승률% | 평균수익% | 손익비(PF) | 평균보유일 | 비교군 평균% | 전반부 평균% | 후반부 평균% | 3종목 집중: 총수익% | 연환산% | 최대낙폭% |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for X in TARGETS:
            rec = per[(N, X)]
            tr = rec["tr"]
            s = stats([t[2] for t in tr], [t[3] for t in tr])
            b = stats([t[1] for t in rec["bs"]], [t[2] for t in rec["bs"]])
            h1 = stats([t[2] for t in tr if t[0] < mid], [])["avg"]
            h2 = stats([t[2] for t in tr if t[0] >= mid], [])["avg"]
            pf = portfolio(tr) if tr else dict(total=np.nan, cagr=np.nan, mdd=np.nan)
            L.append(f"| +{X}% | {s['n']} | {fmt(s['win'])} | {fmt(s['avg'], 2)} | {fmt(s['pf'], 2)} | {fmt(s['hold'])} | {fmt(b['avg'], 2)} | {fmt(h1, 2)} | {fmt(h2, 2)} | {fmt(pf['total'])} | {fmt(pf['cagr'])} | {fmt(pf['mdd'])} |")
        L.append("")
    L.append("## 읽는 법\n")
    L.append("- **평균수익%**: 한 번 거래할 때 비용을 뺀 평균. 0보다 작으면 이 규칙은 돈을 잃는 규칙.")
    L.append("- **비교군 평균%**: 이평선 위 우상향 구간에서 아무 날이나 들어가도 같은 규칙으로 청산했을 때의 평균. 신호 평균이 이보다 뚜렷하게 높아야 '눌림 후 반등'이 의미 있음.")
    L.append("- **전반부/후반부**: 기간을 반으로 나눠 각각 본 평균. 한쪽만 좋고 다른 쪽이 나쁘면 우연일 가능성이 큼.")
    L.append("- **3종목 집중**: 동시에 최대 3종목, 신호가 몰리면 거래대금이 큰 순서로 채움. 실현손익 기준이라 최대낙폭은 실제보다 작게 나옴.")
    return "\n".join(L)


def main():
    t0 = time.time()
    px, oh, val = load()
    per, mid = run(px, oh, val)
    md = report(per, mid, px)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(md)
    S.log(f"완료 {time.time() - t0:.0f}초")
    print(md)


if __name__ == "__main__":
    main()
