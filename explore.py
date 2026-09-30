"""과거 7년에서 '수익이 났던 조건'을 찾아보는 탐색기 (GitHub Actions에서 실행).

방법(과최적화를 막기 위한 장치):
- 미리 정한 단순 규칙 묶음만 테스트한다. (규칙 수와 청산 조합 수를 결과에 그대로 적는다)
- 기간을 반으로 나눠 앞 절반(TRAIN)에서 고른 규칙이 뒤 절반(TEST)에서도 통하는지 본다.
- 앞 절반 상위 규칙이 뒤 절반에서 얼마나 무너지는지도 같이 보여준다.
- 거래 비용 왕복 0.3% 차감. 같은 날 손절/익절이 겹치면 손절로 계산.
한계: 현재 상장 종목만 대상(상장폐지 제외), 종목 필터가 오늘 기준 -> 실제보다 좋게 나오는 쪽으로 편향.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import scan_ma as S
from backtest import load

try:
    from numba import njit
except Exception:  # numba 없으면 느리지만 동작
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

COST = 0.30                 # 왕복 비용 %
MIN_VALUE = 1e9             # 신호일 거래대금 하한
SPLIT = "20230401"          # 이 날짜 전=TRAIN, 이후=TEST
MIN_N = 300                 # 각 구간 최소 거래수
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_result.md")

# (목표수익, 손절, 최대보유일)
EXITS = [
    (0.05, 0.03, 10),
    (0.10, 0.05, 20),
    (0.20, 0.08, 40),
    (9.99, 0.07, 20),    # 목표 없이 20일 보유(손절 -7%)
    (9.99, 0.10, 60),    # 목표 없이 60일 보유(손절 -10%)
]


def exit_name(e):
    t, s, m = e
    return f"{m}일 보유 / 손절-{s*100:g}%" if t > 5 else f"+{t*100:g}% 익절 / 손절-{s*100:g}% / {m}일"


@njit(cache=True)
def run_rule(o, h, l, c, sig_idx, target, stop, maxh):
    n = len(c)
    k = 0
    busy = -1
    ri = np.empty(len(sig_idx), np.int64)
    rr = np.empty(len(sig_idx), np.float64)
    for q in range(len(sig_idx)):
        i = sig_idx[q]
        if i <= busy or i >= n:
            continue
        e = o[i]
        if not e > 0:
            continue
        tp = e * (1 + target)
        sl = e * (1 - stop)
        last = i + maxh - 1
        res = 0.0
        jend = -1
        jmax = min(n - 1, last)
        for j in range(i, jmax + 1):
            if j > i:
                if o[j] <= sl:
                    res = o[j] / e - 1
                    jend = j
                    break
                if o[j] >= tp:
                    res = o[j] / e - 1
                    jend = j
                    break
            if l[j] <= sl:
                res = sl / e - 1
                jend = j
                break
            if h[j] >= tp:
                res = tp / e - 1
                jend = j
                break
        if jend < 0:
            if last < n:
                res = c[last] / e - 1
                jend = last
            else:
                continue
        busy = jend
        ri[k] = i
        rr[k] = res
        k += 1
    return ri[:k], rr[:k]


def rules_for(o, h, l, c, v):
    """한 종목의 규칙별 신호(불리언 배열). 신호일 종가 기준, 진입은 다음날 시가."""
    C = pd.Series(c)
    V = pd.Series(v)
    ma = {n: C.rolling(n).mean() for n in (5, 20, 60, 112, 224)}
    up60 = ma[60] > ma[60].shift(20)
    up224 = ma[224] > ma[224].shift(20)
    align = (ma[60] > ma[112]) & (ma[112] > ma[224])
    vr = V / V.rolling(20).mean().shift(1)                       # 거래대금 배율
    bull = (C > pd.Series(o)) & (C > C.shift(1))
    ret5 = C / C.shift(5) - 1
    hi60, hi120, hi250 = (C.rolling(n).max() for n in (60, 120, 250))
    lo60 = C.rolling(60).min()
    d = C.diff()
    rs = d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    liq = V >= MIN_VALUE
    gap = lambda n: (C / ma[n] - 1).abs()
    cross_up = lambda n: (C > ma[n]) & (C.shift(1) <= ma[n].shift(1))
    gapup = (pd.Series(o) / C.shift(1) - 1) >= 0.03

    R = {
        "비교군: 조건 없이 아무 날": ma[224].notna(),
        "52주(250일) 신고가 돌파": C >= hi250,
        "60일 신고가 + 거래대금 2배": (C >= hi60) & (vr >= 2),
        "120일 신고가 + 정배열": (C >= hi120) & align,
        "224선 상향돌파": cross_up(224),
        "224선 상향돌파 + 거래대금 2배": cross_up(224) & (vr >= 2),
        "112선 상향돌파": cross_up(112),
        "112선 상향돌파 + 거래대금 2배": cross_up(112) & (vr >= 2),
        "60선 상향돌파": cross_up(60),
        "60선 상향돌파 + 거래대금 2배": cross_up(60) & (vr >= 2),
        "224선 위 근접(3%) + 224선 상승": (C > ma[224]) & (gap(224) <= 0.03) & up224,
        "정배열 + 60선 근접(3%) 양봉": align & (gap(60) <= 0.03) & bull,
        "정배열 + 112선 근접(3%) 양봉": align & (gap(112) <= 0.03) & bull,
        "RSI 30 이하 + 양봉": (rsi < 30) & bull,
        "5일 -10% 이상 하락 + 224선 위 + 양봉": (ret5 <= -0.10) & (C > ma[224]) & bull,
        "20선 -10% 이탈 + 양봉": ((C / ma[20] - 1) <= -0.10) & bull,
        "60일 저점 5% 이내 + 양봉": (C <= lo60 * 1.05) & bull,
        "거래대금 3배 + 5% 이상 양봉 + 60선 위": (vr >= 3) & ((C / C.shift(1) - 1) >= 0.05) & (C > ma[60]),
        "갭상승 3% + 거래대금 2배 + 60선 위": gapup & (vr >= 2) & (C > ma[60]),
        "정배열 + 5일선 위 첫 양봉(20선 아래→위)": align & cross_up(20),
    }
    out = {}
    for k, s in R.items():
        sg = (s & liq).fillna(False).values
        out[k] = np.where(sg[:-1])[0] + 1
    return out


def collect(px, oh, val):
    dates = np.array(px.index)
    trades = {}     # (rule, exit_idx) -> list of arrays (date_idx_as_str, ret%)
    t0 = time.time()
    for k, code in enumerate(px.columns):
        c0 = px[code].values
        ok = ~np.isnan(c0)
        if ok.sum() < 500:
            continue
        d = dates[ok]
        o, h, l, c = (np.ascontiguousarray(oh["o"][code].values[ok]), np.ascontiguousarray(oh["h"][code].values[ok]),
                      np.ascontiguousarray(oh["l"][code].values[ok]), np.ascontiguousarray(c0[ok]))
        v = np.nan_to_num(val[code].values[ok])
        sigs = rules_for(o, h, l, c, v)
        for name, idx in sigs.items():
            if len(idx) == 0:
                continue
            for ei, (t, s, m) in enumerate(EXITS):
                ri, rr = run_rule(o, h, l, c, idx.astype(np.int64), t, s, m)
                if len(ri):
                    trades.setdefault((name, ei), []).append((d[ri], rr * 100 - COST))
        if k % 100 == 0:
            S.log(f"  탐색 {k}/{px.shape[1]} ({time.time() - t0:.0f}초)")
    merged = {}
    for key, lst in trades.items():
        merged[key] = (np.concatenate([a for a, _ in lst]), np.concatenate([b for _, b in lst]))
    return merged


def stat(r):
    if len(r) == 0:
        return dict(n=0, avg=np.nan, win=np.nan, pf=np.nan)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), avg=r.mean(), win=(r > 0).mean() * 100, pf=(pos / neg if neg > 0 else np.inf))


def f(x, p=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("inf" if x == np.inf else f"{x:.{p}f}")


def analyze(merged):
    rows = []
    for (name, ei), (dts, rets) in merged.items():
        tr = rets[dts < SPLIT]
        te = rets[dts >= SPLIT]
        yrs = pd.Series(rets).groupby(pd.Series([x[:4] for x in dts]).values).mean()
        a, b = stat(tr), stat(te)
        rows.append(dict(rule=name, exit=exit_name(EXITS[ei]), ei=ei, ntr=a["n"], atr=a["avg"], nte=b["n"], ate=b["avg"],
                         wte=b["win"], pfte=b["pf"], ypos=int((yrs > 0).sum()), ytot=len(yrs)))
    return pd.DataFrame(rows)


def report(df, px):
    L = ["# 과거 데이터에서 수익이 났던 조건 탐색\n"]
    L.append(f"- 기간 {px.index[0]}~{px.index[-1]}, {px.shape[1]}종목. TRAIN={px.index[0]}~{SPLIT} 이전 / TEST={SPLIT} 이후")
    L.append(f"- 테스트한 조합: 규칙 {df['rule'].nunique()}개 x 청산 {len(EXITS)}개 = {len(df)}개 (많이 볼수록 우연히 좋은 게 섞임)")
    L.append(f"- 비용 왕복 {COST}% 차감, 종목당 동시 1포지션, 같은 날 손절/익절 겹치면 손절.")
    L.append("- 편향: 상장폐지 종목이 빠져 있고 종목 필터가 오늘 기준이라 결과가 실제보다 좋게 나옴.\n")

    L.append("## 비교군 (조건 없이 아무 날 사서 같은 방식으로 청산)\n")
    L.append("| 청산 | TRAIN 평균% | TEST 평균% | TEST 승률% |")
    L.append("|---|---|---|---|")
    for _, r in df[df.rule.str.startswith("비교군")].sort_values("ei").iterrows():
        L.append(f"| {r['exit']} | {f(r['atr'])} | {f(r['ate'])} | {f(r['wte'], 1)} |")
    L.append("")

    base = df[~df.rule.str.startswith("비교군")].copy()
    ok = base[(base.ntr >= MIN_N) & (base.nte >= MIN_N)]
    L.append("## 1) 앞 절반(TRAIN) 성적 상위 10개가 뒤 절반(TEST)에서 어떻게 됐나\n")
    top = ok.sort_values("atr", ascending=False).head(10)
    L.append("| 규칙 | 청산 | TRAIN 거래수 | TRAIN 평균% | TEST 거래수 | TEST 평균% |")
    L.append("|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        L.append(f"| {r['rule']} | {r['exit']} | {r['ntr']} | {f(r['atr'])} | {r['nte']} | {f(r['ate'])} |")
    if len(top):
        L.append(f"\n- TRAIN 상위 10개의 평균: TRAIN {f(top.atr.mean())}% -> TEST {f(top.ate.mean())}%. 차이가 클수록 '과거에만 맞은' 규칙이라는 뜻.\n")

    L.append("## 2) 앞 절반과 뒤 절반 모두 플러스였던 규칙 (거래수 각 %d 이상)\n" % MIN_N)
    bl = df[df.rule.str.startswith("비교군")].set_index("ei")["ate"]
    ok = ok.assign(edge=ok.ate - ok.ei.map(bl))
    both = ok[(ok.atr > 0) & (ok.ate > 0)].sort_values("edge", ascending=False)
    L.append(f"- 해당 {len(both)}개 / 검토 대상 {len(ok)}개. 비교군 대비 %p = 같은 청산 방식으로 아무 날이나 샀을 때(TEST)보다 얼마나 더 나은지. 이게 작으면 규칙이 아니라 시장 상승 덕분.\n")
    L.append("| 규칙 | 청산 | TRAIN 평균% | TEST 거래수 | TEST 평균% | 비교군 대비 %p | TEST 승률% | TEST 손익비 | 플러스 연도 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in both.head(20).iterrows():
        L.append(f"| {r['rule']} | {r['exit']} | {f(r['atr'])} | {r['nte']} | {f(r['ate'])} | {f(r['edge'])} | {f(r['wte'], 1)} | {f(r['pfte'])} | {r['ypos']}/{r['ytot']} |")
    L.append("")

    L.append("## 3) 뒤 절반(TEST)에서 평균이 가장 나빴던 규칙 5개 (피해야 할 것)\n")
    bad = ok.sort_values("ate").head(5)
    L.append("| 규칙 | 청산 | TEST 평균% | TEST 승률% |")
    L.append("|---|---|---|---|")
    for _, r in bad.iterrows():
        L.append(f"| {r['rule']} | {r['exit']} | {f(r['ate'])} | {f(r['wte'], 1)} |")
    L.append("")

    L.append("## 읽는 법\n")
    L.append("- 평균%는 한 번 거래당 비용 차감 후 평균. 규칙 대부분이 비교군과 별 차이 없거나 마이너스면 '이평선/거래량 신호가 통하지 않는다'는 뜻.")
    L.append("- 2번 표에 나온 것도 우연일 수 있음. 가장 믿을 만한 건 TRAIN·TEST 둘 다 플러스이고, 플러스 연도가 많고, 비교군보다 뚜렷하게 높은 규칙.")
    L.append("- 이 표는 '평균'이라 실제 계좌의 최대낙폭, 동시 보유 종목 수 제한, 체결 문제는 반영 안 됨.")
    return "\n".join(L)


def main():
    t0 = time.time()
    px, oh, val = load()
    merged = collect(px, oh, val)
    df = analyze(merged)
    df.to_csv(os.path.join(os.path.dirname(OUT), "explore_all.csv"), index=False, encoding="utf-8-sig")
    md = report(df, px)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(md)
    S.log(f"완료 {time.time() - t0:.0f}초")
    print(md)


if __name__ == "__main__":
    main()
