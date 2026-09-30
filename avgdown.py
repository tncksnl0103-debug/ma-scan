"""물타기 전략 비교 백테스트 (GitHub Actions에서 실행)

같은 진입 신호(explore.py의 규칙)에서 세 가지 운용을 비교한다. 자본 1(=10만원) 기준 수익률.
 A) 한 번에 전액 매수, 진입가 -10% 손절, 최대 60거래일 보유
 B) 절반 매수 -> 진입가 -10%에서 절반 추가 매수(물타기) -> 평균단가 -10%에서 손절, 최대 60거래일
 C) 절반 매수 -> 진입가 -10%에서 절반 추가 매수 -> 추가 매수가 -10%에서 손절, 최대 60거래일
 (추가 매수가 안 나가면 나머지 절반은 놀리는 현금으로 계산: 수익률 분모는 항상 전체 자본)
같은 날 추가 매수와 손절이 겹치면 손절로 계산. 갭하락이면 시가에 체결. 왕복 비용 0.3% (투입 금액 기준).
"""
import os
import time

import numpy as np
import pandas as pd

import scan_ma as S
from backtest import load
import explore as E

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

MAXH = 60
DROP = 0.10
STOP = 0.10
COST = 0.003
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avgdown_result.md")
RULES = [
    "정배열 + 60선 근접(3%) 양봉",
    "정배열 + 112선 근접(3%) 양봉",
    "52주(250일) 신고가 돌파",
    "112선 상향돌파",
    "60선 상향돌파",
]


@njit(cache=True)
def sim_one(o, h, l, c, i, mode, maxh, drop, stop):
    """mode 0=A 전액, 1=B 평균단가 손절, 2=C 추가매수가 손절. 반환 (수익률(자본대비), 추가매수여부, 손절여부, 종료idx)"""
    n = len(c)
    e = o[i]
    if not e > 0:
        return np.nan, 0, 0, i
    last = min(n - 1, i + maxh - 1)
    if i + maxh - 1 > n - 1:
        return np.nan, 0, 0, i          # 아직 60일 안 지난 거래는 제외
    if mode == 0:
        sl = e * (1 - stop)
        for j in range(i, last + 1):
            if j > i and o[j] <= sl:
                return (o[j] / e - 1) - COST, 0, 1, j
            if l[j] <= sl:
                return (sl / e - 1) - COST, 0, 1, j
        return (c[last] / e - 1) - COST, 0, 0, last
    sh1 = 0.5 / e
    invested = 0.5
    sh2 = 0.0
    added = 0
    lvl = e * (1 - drop)
    sl = 0.0
    for j in range(i, last + 1):
        if added == 0:
            if l[j] <= lvl:
                px = lvl
                if j > i and o[j] <= lvl:
                    px = o[j]
                sh2 = 0.5 / px
                invested = 1.0
                added = 1
                avg = invested / (sh1 + sh2)
                if mode == 1:
                    sl = avg * (1 - stop)
                else:
                    sl = px * (1 - stop)
                # 같은 날 손절 겹침(보수적)
                if l[j] <= sl:
                    x = sl if o[j] > sl else o[j]
                    pnl = sh1 * (x - e) + sh2 * (x - px)
                    return pnl - COST * invested, 1, 1, j
        else:
            if o[j] <= sl:
                x = o[j]
                pnl = sh1 * (x - e) + sh2 * (x - lvl)
                return pnl - COST * invested, 1, 1, j
            if l[j] <= sl:
                x = sl
                pnl = sh1 * (x - e) + sh2 * (x - lvl)
                return pnl - COST * invested, 1, 1, j
    x = c[last]
    pnl = sh1 * (x - e)
    if added == 1:
        pnl += sh2 * (x - lvl)
    return pnl - COST * invested, added, 0, last


def collect(px, oh, val):
    dates = np.array(px.index)
    out = {}   # (rule, mode) -> list of (date, ret, added, stopped)
    for code in px.columns:
        c0 = px[code].values
        ok = ~np.isnan(c0)
        if ok.sum() < 500:
            continue
        d = dates[ok]
        o, h, l, c = (np.ascontiguousarray(oh["o"][code].values[ok]), np.ascontiguousarray(oh["h"][code].values[ok]),
                      np.ascontiguousarray(oh["l"][code].values[ok]), np.ascontiguousarray(c0[ok]))
        v = np.nan_to_num(val[code].values[ok])
        sigs = E.rules_for(o, h, l, c, v)
        for name in RULES:
            idx = sigs.get(name)
            if idx is None or len(idx) == 0:
                continue
            for mode in (0, 1, 2):
                busy = -1
                for i in idx:
                    if i <= busy or i >= len(c):
                        continue
                    r, added, stopped, j = sim_one(o, h, l, c, int(i), mode, MAXH, DROP, STOP)
                    if r != r:
                        continue
                    busy = j
                    out.setdefault((name, mode), []).append((d[i], r * 100, added, stopped))
    return out


def summarize(rows):
    if not rows:
        return None
    a = np.array([(r[1], r[2], r[3]) for r in rows], dtype=float)
    r = a[:, 0]
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), avg=r.mean(), win=(r > 0).mean() * 100, pf=(pos / neg if neg > 0 else np.inf),
                worst=r.min(), p5=np.percentile(r, 5), add=a[:, 1].mean() * 100, stop=a[:, 2].mean() * 100)


def fm(x, p=2):
    return "-" if x is None or x != x else ("inf" if x == np.inf else f"{x:.{p}f}")


def report(res, px):
    names = {0: "A 전액 매수 / -10% 손절", 1: "B 절반+물타기 / 평균단가 -10% 손절", 2: "C 절반+물타기 / 추가매수가 -10% 손절"}
    L = ["# 물타기(-10%에서 절반 추가 매수) 전략 비교\n"]
    L.append(f"- 기간 {px.index[0]}~{px.index[-1]}, {px.shape[1]}종목. 자본 100 기준 수익률(%). 최대 60거래일 보유, 비용 0.3%(투입액 기준).")
    L.append("- 물타기가 안 나간 거래는 나머지 절반 자본이 놀고 있는 것으로 계산(분모는 항상 전체 자본).")
    L.append("- 편향: 상장폐지 종목 없음, 종목 필터가 오늘 기준 -> 실제보다 좋게 나옴. 특히 '떨어지면 더 산다'는 방식이 이 편향에 유리함.\n")
    for period, cond in (("전체", lambda d: True), ("앞 절반(~2023-03)", lambda d: d < E.SPLIT), ("뒤 절반(2023-04~)", lambda d: d >= E.SPLIT)):
        L.append(f"## {period}\n")
        L.append("| 진입 규칙 | 운용 | 거래수 | 평균% | 승률% | 손익비 | 최악 거래% | 하위5% 거래% | 물타기 발생% | 손절 발생% |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for name in RULES:
            for mode in (0, 1, 2):
                rows = [r for r in res.get((name, mode), []) if cond(r[0])]
                s = summarize(rows)
                if not s:
                    continue
                L.append(f"| {name} | {names[mode]} | {s['n']} | {fm(s['avg'])} | {fm(s['win'], 1)} | {fm(s['pf'])} | {fm(s['worst'], 1)} | {fm(s['p5'], 1)} | {fm(s['add'], 1)} | {fm(s['stop'], 1)} |")
        L.append("")
    L.append("## 읽는 법\n")
    L.append("- 평균%가 A보다 낮고 최악 거래가 더 깊으면, 물타기가 위험만 키우고 수익은 못 늘린 것.")
    L.append("- 물타기 발생%: 진입 후 -10%까지 내려가 추가 매수가 나간 비율. 손절 발생%는 최종적으로 손절로 끝난 비율.")
    return "\n".join(L)


def main():
    t0 = time.time()
    px, oh, val = load()
    res = collect(px, oh, val)
    md = report(res, px)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(md)
    S.log(f"완료 {time.time() - t0:.0f}초")
    print(md)


if __name__ == "__main__":
    main()
