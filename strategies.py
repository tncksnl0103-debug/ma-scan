"""널리 알려진 매매 전략 22개를 같은 조건으로 백테스트 + 불타기(오르면 추가 매수) 비교 (GitHub Actions에서 실행)

- 신호는 종가 기준으로 확인하고 다음날 시가에 진입한다. (변동성 돌파만 장중 돌파가에 진입)
- 청산은 각 전략의 자체 규칙(지표 조건 -> 다음날 시가, 시간 청산은 그날 종가), 손절은 전략별로 지정(장중 저가 기준).
- 왕복 비용 0.3% 차감. 종목당 동시 1포지션. 신호일 거래대금 10억 이상만.
- '시장 평균 대비'는 같은 기간 아무 종목을 같은 일수만큼 들고 있었을 때 평균(같은 거래일 필터, 비용 동일)을 뺀 값.
- 앞/뒤 절반(TRAIN/TEST)을 나눠서 본다.
한계: 상장폐지 종목이 없고 종목 필터가 오늘 기준이라 실제보다 좋게 나오는 쪽으로 편향됨.
"""
import os
import time

import numpy as np
import pandas as pd

import scan_ma as S
from backtest import load

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

COST = 0.30
MIN_VALUE = 1e9
SPLIT = "20230401"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies_result.md")
HS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 90, 120, 180, 250])


@njit(cache=True)
def run_generic(o, h, l, c, sig_idx, ex, stop, maxh):
    n = len(c)
    busy = -1
    k = 0
    ri = np.empty(len(sig_idx), np.int64)
    rr = np.empty(len(sig_idx), np.float64)
    rh = np.empty(len(sig_idx), np.int64)
    for q in range(len(sig_idx)):
        i = sig_idx[q]
        if i <= busy or i >= n - 1:
            continue
        e = o[i]
        if not e > 0:
            continue
        sl = e * (1 - stop) if stop > 0 else -1.0
        res = 0.0
        jend = -1
        done = False
        last = i + maxh - 1
        for j in range(i, n):
            if stop > 0:
                if j > i and o[j] <= sl:
                    res = o[j] / e - 1
                    jend = j
                    done = True
                    break
                if l[j] <= sl:
                    res = sl / e - 1
                    jend = j
                    done = True
                    break
            if j >= last:
                res = c[j] / e - 1
                jend = j
                done = True
                break
            if ex[j]:
                if j + 1 < n:
                    res = o[j + 1] / e - 1
                    jend = j + 1
                    done = True
                break
        if not done:
            continue
        busy = jend
        ri[k] = i
        rr[k] = res
        rh[k] = jend - i
        k += 1
    return ri[:k], rr[:k], rh[:k]


@njit(cache=True)
def run_vb(o, h, l, c, kk, filt):
    """변동성 돌파: 오늘 시가 + kk*(어제 고-저) 돌파가에 매수, 다음날 시가에 매도."""
    n = len(c)
    ri = np.empty(n, np.int64)
    rr = np.empty(n, np.float64)
    m = 0
    for t in range(1, n - 1):
        if not filt[t - 1]:
            continue
        rg = h[t - 1] - l[t - 1]
        if not rg > 0 or not o[t] > 0:
            continue
        trig = o[t] + kk * rg
        if h[t] >= trig and o[t + 1] > 0:
            ri[m] = t
            rr[m] = o[t + 1] / trig - 1
            m += 1
    return ri[:m], rr[:m]


@njit(cache=True)
def run_pyramid(o, h, l, c, sig_idx, mode, maxh, add_at, stop):
    """mode 0=전액 처음부터, 1=절반 진입 + add_at 상승 시 절반 추가(손절은 최초 진입가 기준 유지),
    2=절반 진입 + 추가(손절은 평균단가 기준으로 재설정). 수익률은 항상 전체 자본(1) 기준."""
    n = len(c)
    busy = -1
    k = 0
    ri = np.empty(len(sig_idx), np.int64)
    rr = np.empty(len(sig_idx), np.float64)
    ad = np.empty(len(sig_idx), np.int64)
    for q in range(len(sig_idx)):
        i = sig_idx[q]
        if i <= busy or i + maxh - 1 > n - 1:
            continue
        e = o[i]
        if not e > 0:
            continue
        last = i + maxh - 1
        frac = 1.0 if mode == 0 else 0.5
        sh1 = frac / e
        sh2 = 0.0
        px2 = 0.0
        added = 0
        sl = e * (1 - stop)
        lvl = e * (1 + add_at)
        invested = frac
        res = 0.0
        jend = -1
        for j in range(i, last + 1):
            # 손절 (장중, 갭이면 시가)
            x = -1.0
            if j > i and o[j] <= sl:
                x = o[j]
            elif l[j] <= sl:
                x = sl
            if x > 0:
                res = sh1 * (x - e) + sh2 * (x - px2) - COST / 100.0 * invested
                jend = j
                break
            if mode != 0 and added == 0 and h[j] >= lvl:
                px2 = lvl if (j == i or o[j] < lvl) else o[j]
                sh2 = 0.5 / px2
                invested = 1.0
                added = 1
                if mode == 2:
                    sl = (invested / (sh1 + sh2)) * (1 - stop)
        if jend < 0:
            x = c[last]
            res = sh1 * (x - e) + sh2 * (x - px2) - COST / 100.0 * invested
            jend = last
        busy = jend
        ri[k] = i
        rr[k] = res
        ad[k] = added
        k += 1
    return ri[:k], rr[:k], ad[:k]


def rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


STRATS = [
    # (이름, 매매 방법 요약, stop, maxh)
    ("01 터틀 20일 돌파", "20일 고가 돌파 종가 확인 -> 다음날 매수 / 10일 저가 이탈 시 매도 / 손절 -10%", 0.10, 250),
    ("02 터틀 55일 돌파", "55일 고가 돌파 -> 매수 / 20일 저가 이탈 시 매도 / 손절 -10%", 0.10, 250),
    ("03 골든크로스 50/200", "50일선이 200일선 상향 돌파 -> 매수 / 다시 하향 돌파 시 매도 / 손절 -15%", 0.15, 250),
    ("04 골든크로스 20/60", "20일선이 60일선 상향 돌파 -> 매수 / 하향 돌파 시 매도 / 손절 -10%", 0.10, 250),
    ("05 60일선 종가 돌파", "종가가 60일선을 아래에서 위로 돌파 -> 매수 / 종가가 60일선 아래로 내려가면 매도 / 손절 -10%", 0.10, 120),
    ("06 224일선 종가 돌파", "종가가 224일선 상향 돌파 -> 매수 / 종가가 224일선 아래로 내려가면 매도 / 손절 -10%", 0.10, 250),
    ("07 정배열 + 60선 눌림 양봉", "60>112>224 정배열에서 60일선 3% 이내까지 눌린 양봉 -> 매수 / 60거래일 보유 / 손절 -10%", 0.10, 60),
    ("08 52주 신고가 + 거래대금 2배", "250일 고가 돌파 + 거래대금 평소 2배 -> 매수 / 20일 저가 이탈 시 매도 / 손절 -10%", 0.10, 120),
    ("09 볼린저 상단 돌파(추세)", "종가가 볼린저 상단(20,2) 돌파 + 거래대금 1.5배 -> 매수 / 종가가 20일선 아래면 매도 / 손절 -10%", 0.10, 60),
    ("10 볼린저 하단 이탈 후 복귀(반등)", "전날 종가가 하단 밖 -> 오늘 안으로 복귀 -> 매수 / 20일선 도달 시 매도 / 손절 -10% / 최대 20일", 0.10, 20),
    ("11 볼린저 스퀴즈 돌파", "밴드 폭이 120일 중 하위 10%로 좁았다가 상단 돌파 -> 매수 / 20일선 아래면 매도 / 손절 -10%", 0.10, 60),
    ("12 RSI(2) 과매도 (커너스)", "RSI(2)<5 + 종가>100일선 -> 매수 / RSI(2)>30이면 매도 / 손절 없음 / 최대 10일", 0.0, 10),
    ("13 Double 7s", "종가>200일선 + 종가가 7일 최저 -> 매수 / 종가가 7일 최고면 매도 / 손절 없음 / 최대 20일", 0.0, 20),
    ("14 RSI(14) 30 상향 복귀", "RSI(14)가 30 아래에서 30 위로 복귀 -> 매수 / RSI 50 도달 시 매도 / 손절 -10% / 최대 30일", 0.10, 30),
    ("15 이격도 90 이하 반등", "종가가 20일선의 90% 이하 + 양봉 -> 매수 / 20일선 도달 시 매도 / 손절 -10% / 최대 20일", 0.10, 20),
    ("16 변동성 돌파 (래리 윌리엄스)", "당일 시가 + 0.5x(전일 고-저) 돌파 시 매수 -> 다음날 시가에 매도 (손절 없음)", 0.0, 1),
    ("17 변동성 돌파 + 5일선 위", "위와 같고 전날 종가가 5일선 위일 때만 (손절 없음)", 0.0, 1),
    ("18 3일 연속 하락 후 반등", "3일 연속 하락 + 종가>200일선 -> 매수 / 3일 보유 / 손절 없음", 0.0, 3),
    ("19 MACD 골든크로스", "MACD(12,26,9)가 0 아래에서 시그널선 상향 + 종가>60일선 -> 매수 / 시그널선 하향 시 매도 / 손절 -10%", 0.10, 60),
    ("20 일목 구름 돌파", "종가가 구름대 위로 돌파 -> 매수 / 종가가 기준선(26일) 아래면 매도 / 손절 -10%", 0.10, 120),
    ("21 스토캐스틱 과매도 교차", "슬로우 %K가 20 아래에서 %D 상향 교차 -> 매수 / %K>80이면 매도 / 손절 -10% / 최대 30일", 0.10, 30),
    ("22 20일선 눌림목", "상승추세(종가>60선, 20선>60선)에서 20일선 2% 이내 양봉 -> 매수 / 종가가 20일선 -3% 아래면 매도 / 손절 -8%", 0.08, 40),
]


def signals(o, h, l, c, v):
    C, H, L, O, V = (pd.Series(x) for x in (c, h, l, o, v))
    ma = {n: C.rolling(n).mean() for n in (5, 20, 50, 60, 100, 112, 200, 224)}
    liq = (V >= MIN_VALUE)
    vr = V / V.rolling(20).mean().shift(1)
    bull = (C > O) & (C > C.shift(1))
    cross_up = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
    hh = lambda n: H.rolling(n).max().shift(1)
    ll = lambda n: L.rolling(n).min().shift(1)
    sd = C.rolling(20).std()
    bb_up, bb_lo = ma[20] + 2 * sd, ma[20] - 2 * sd
    bw = (bb_up - bb_lo) / ma[20]
    r2, r14 = rsi(C, 2), rsi(C, 14)
    e12, e26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig9 = macd.ewm(span=9, adjust=False).mean()
    tenkan = (H.rolling(9).max() + L.rolling(9).min()) / 2
    kijun = (H.rolling(26).max() + L.rolling(26).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(26)
    spb = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26)
    ctop, cbot = np.maximum(spa, spb), np.minimum(spa, spb)
    fk = (C - L.rolling(14).min()) / (H.rolling(14).max() - L.rolling(14).min() + 1e-12) * 100
    sk = fk.rolling(3).mean()
    sdd = sk.rolling(3).mean()
    align = (ma[60] > ma[112]) & (ma[112] > ma[224])

    S_ = {}
    X_ = {}
    S_["01"] = C > hh(20);                                   X_["01"] = C < ll(10)
    S_["02"] = C > hh(55);                                   X_["02"] = C < ll(20)
    S_["03"] = cross_up(ma[50], ma[200]);                    X_["03"] = ma[50] < ma[200]
    S_["04"] = cross_up(ma[20], ma[60]);                     X_["04"] = ma[20] < ma[60]
    S_["05"] = cross_up(C, ma[60]);                          X_["05"] = C < ma[60]
    S_["06"] = cross_up(C, ma[224]);                         X_["06"] = C < ma[224]
    S_["07"] = align & ((C / ma[60] - 1).abs() <= 0.03) & bull; X_["07"] = pd.Series(False, index=C.index)
    S_["08"] = (C > hh(250)) & (vr >= 2);                    X_["08"] = C < ll(20)
    S_["09"] = (C > bb_up) & (C.shift(1) <= bb_up.shift(1)) & (vr >= 1.5); X_["09"] = C < ma[20]
    S_["10"] = (C.shift(1) < bb_lo.shift(1)) & (C >= bb_lo);  X_["10"] = C >= ma[20]
    S_["11"] = (bw.shift(1) <= bw.rolling(120).quantile(0.10).shift(1)) & (C > bb_up); X_["11"] = C < ma[20]
    S_["12"] = (r2 < 5) & (C > ma[100]);                     X_["12"] = r2 > 30
    S_["13"] = (C > ma[200]) & (C <= C.rolling(7).min());    X_["13"] = C >= C.rolling(7).max()
    S_["14"] = (r14.shift(1) < 30) & (r14 >= 30);            X_["14"] = r14 > 50
    S_["15"] = ((C / ma[20]) <= 0.90) & bull;                X_["15"] = C >= ma[20]
    S_["18"] = (C < C.shift(1)) & (C.shift(1) < C.shift(2)) & (C.shift(2) < C.shift(3)) & (C > ma[200]); X_["18"] = pd.Series(False, index=C.index)
    S_["19"] = cross_up(macd, sig9) & (macd < 0) & (C > ma[60]); X_["19"] = macd < sig9
    S_["20"] = (C > ctop) & (C.shift(1) <= ctop.shift(1));   X_["20"] = C < kijun
    S_["21"] = cross_up(sk, sdd) & (sk < 20);                X_["21"] = sk > 80
    S_["22"] = (C > ma[60]) & (ma[20] > ma[60]) & ((C / ma[20] - 1).abs() <= 0.02) & bull; X_["22"] = C < ma[20] * 0.97
    vb1 = liq.values
    vb2 = (liq & (C > ma[5])).fillna(False).values
    out = {}
    for key in S_:
        sg = (S_[key] & liq).fillna(False).values
        out[key] = (np.where(sg[:-1])[0] + 1, X_[key].fillna(False).values.astype(np.bool_))
    return out, vb1, vb2


def market_base(o, c, liq, part_mask_train):
    """보유일 H별 시장 평균 (매수: 다음날 시가~H일째 종가). liq는 신호일(전날) 거래대금 조건."""
    n = len(c)
    res = {}
    for hi_, H in enumerate(HS):
        if n <= H + 2:
            continue
        idx = np.arange(1, n - H + 1)
        m = liq[idx - 1] & (o[idx] > 0)
        r = c[idx + H - 1] / o[idx] - 1
        for per, sel in (("tr", part_mask_train[idx]), ("te", ~part_mask_train[idx])):
            mm = m & sel
            res[(per, hi_)] = (r[mm].sum(), int(mm.sum()))
    # 1일 시가->다음날 시가 (변동성 돌파 비교용)
    idx = np.arange(1, n - 1)
    m = liq[idx - 1] & (o[idx] > 0) & (o[idx + 1] > 0)
    r = o[idx + 1] / o[idx] - 1
    for per, sel in (("tr", part_mask_train[idx]), ("te", ~part_mask_train[idx])):
        mm = m & sel
        res[(per, "vb")] = (r[mm].sum(), int(mm.sum()))
    return res


def collect(px, oh, val):
    dates = np.array(px.index)
    trades = {}      # key -> list of (dates array, ret% array, hold array)
    pyr = {}
    base_sum = {}
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
        sg, vb1, vb2 = signals(o, h, l, c, v)
        liq = v >= MIN_VALUE
        tr_mask = d < SPLIT
        for key, (b, s) in sg.items():
            name = [s_ for s_ in STRATS if s_[0].startswith(key)][0]
            _, _, stop, maxh = name
            ri, rr, rh = run_generic(o, h, l, c, b.astype(np.int64), s, stop, maxh)
            if len(ri):
                trades.setdefault(key, []).append((d[ri], rr * 100 - COST, rh))
        for key, filt in (("16", vb1), ("17", vb2)):
            ri, rr = run_vb(o, h, l, c, 0.5, filt)
            if len(ri):
                trades.setdefault(key, []).append((d[ri], rr * 100 - COST, np.ones(len(ri), np.int64)))
        # 불타기 비교 (07번 진입 신호)
        b07 = sg["07"][0].astype(np.int64)
        for mode in (0, 1, 2):
            ri, rr, ad = run_pyramid(o, h, l, c, b07, mode, 60, 0.10, 0.10)
            if len(ri):
                pyr.setdefault(mode, []).append((d[ri], rr * 100, ad))
        mb = market_base(o, c, liq, tr_mask)
        for kk, (s_, n_) in mb.items():
            a = base_sum.get(kk, (0.0, 0))
            base_sum[kk] = (a[0] + s_, a[1] + n_)
        if k % 150 == 0:
            S.log(f"  진행 {k}/{px.shape[1]} ({time.time() - t0:.0f}초)")
    merged = {kk: tuple(np.concatenate([x[j] for x in lst]) for j in range(3)) for kk, lst in trades.items()}
    pm = {kk: tuple(np.concatenate([x[j] for x in lst]) for j in range(3)) for kk, lst in pyr.items()}
    base = {kk: (s_ / n_ * 100 - COST if n_ else np.nan) for kk, (s_, n_) in base_sum.items()}
    return merged, pm, base


def base_at(base, per, H):
    xs, ys = [], []
    for hi_, hh_ in enumerate(HS):
        y = base.get((per, hi_))
        if y is not None and y == y:
            xs.append(hh_)
            ys.append(y)
    return np.interp(np.clip(H + 1, 1, HS[-1]), xs, ys)   # 보유일 H -> 진입일 포함 H+1일 근사


def stat_row(rets):
    r = np.asarray(rets, float)
    if len(r) == 0:
        return dict(n=0, avg=np.nan, win=np.nan, pf=np.nan)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), avg=r.mean(), win=(r > 0).mean() * 100, pf=(pos / neg if neg > 0 else np.inf))


def f(x, p=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("inf" if x == np.inf else f"{x:.{p}f}")


def analyze(merged, base, nyears):
    rows = []
    for key, (dts, rets, hold) in merged.items():
        name = [s_ for s_ in STRATS if s_[0].startswith(key)][0]
        tr = dts < SPLIT
        if key in ("16", "17"):
            bt, be = base[("tr", "vb")], base[("te", "vb")]
            exc = np.where(tr, rets - bt, rets - be)
        else:
            exc = np.where(tr, rets - base_at(base, "tr", hold), rets - base_at(base, "te", hold))
        yrs = pd.Series(rets).groupby(np.array([x[:4] for x in dts])).mean()
        s_all, s_tr, s_te = stat_row(rets), stat_row(rets[tr]), stat_row(rets[~tr])
        rows.append(dict(name=name[0], how=name[1], n=s_all["n"], per_year=s_all["n"] / nyears, hold=hold.mean(),
                         avg=s_all["avg"], win=s_all["win"], pf=s_all["pf"], exc=exc.mean(),
                         atr=s_tr["avg"], ate=s_te["avg"], etr=exc[tr].mean() if tr.any() else np.nan,
                         ete=exc[~tr].mean() if (~tr).any() else np.nan, ypos=int((yrs > 0).sum()), ytot=len(yrs),
                         p5=np.percentile(rets, 5), worst=rets.min()))
    return pd.DataFrame(rows).sort_values("name")


def report(df, pm, base, px):
    L = ["# 널리 알려진 매매 전략 22개 백테스트\n"]
    L.append(f"- 기간 {px.index[0]}~{px.index[-1]}, {px.shape[1]}종목(오늘 기준 시총 500억+, 거래대금 10억+, ETF/ETN 제외). TRAIN=~{SPLIT} 이전, TEST=이후")
    L.append(f"- 왕복 비용 {COST}% 차감. 종목당 동시 1포지션. 평균%는 거래 1건당. '시장 대비'는 같은 기간·같은 보유일 아무 종목 평균을 뺀 값(%p).")
    L.append("- 편향: 상장폐지 종목 없음 + 종목 필터가 오늘 기준 -> 실제보다 좋게 나옴.\n")
    L.append("## 전체 표\n")
    L.append("| 전략 | 거래수 | 연간 거래수 | 평균보유일 | 평균% | 승률% | 손익비 | 시장 대비 %p | TRAIN 평균% | TEST 평균% | TRAIN 시장대비 | TEST 시장대비 | 플러스 연도 | 하위5% 거래% | 최악 거래% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L.append(f"| {r['name']} | {r['n']} | {f(r['per_year'], 0)} | {f(r['hold'], 1)} | {f(r['avg'])} | {f(r['win'], 1)} | {f(r['pf'])} | {f(r['exc'])} | {f(r['atr'])} | {f(r['ate'])} | {f(r['etr'])} | {f(r['ete'])} | {r['ypos']}/{r['ytot']} | {f(r['p5'], 1)} | {f(r['worst'], 1)} |")
    L.append("\n## 매매 방법\n")
    for _, r in df.iterrows():
        L.append(f"- **{r['name']}**: {r['how']}")
    L.append("\n## 불타기 비교 (07번 진입 신호, 60거래일, 자본 1 기준)\n")
    L.append("| 운용 | 거래수 | 평균% | 승률% | 손익비 | 하위5% 거래% | 최악 거래% | 추가매수 발생% |")
    L.append("|---|---|---|---|---|---|---|---|")
    names = {0: "A 처음부터 전액 매수 / -10% 손절", 1: "P1 절반 매수 -> +10%에서 절반 추가 / 손절은 최초가 -10% 유지", 2: "P2 절반 매수 -> +10%에서 절반 추가 / 손절은 평균단가 -10%"}
    for m in (0, 1, 2):
        if m not in pm:
            continue
        dts, rets, ad = pm[m]
        s_ = stat_row(rets)
        L.append(f"| {names[m]} | {s_['n']} | {f(s_['avg'])} | {f(s_['win'], 1)} | {f(s_['pf'])} | {f(np.percentile(rets, 5), 1)} | {f(rets.min(), 1)} | {f(ad.mean() * 100, 1)} |")
    L.append("")
    for m in (0, 1, 2):
        if m in pm:
            dts, rets, ad = pm[m]
            tr = dts < SPLIT
            L.append(f"- {names[m]}: TRAIN 평균 {f(rets[tr].mean())}% / TEST 평균 {f(rets[~tr].mean())}%")
    L.append("\n## 시장 평균 (참고: 거래일 필터 통과한 아무 종목, 보유일별 평균%, 비용 차감)\n")
    L.append("| 보유일 | TRAIN | TEST |")
    L.append("|---|---|---|")
    for hi_, H in enumerate(HS):
        L.append(f"| {H} | {f(base.get(('tr', hi_)))} | {f(base.get(('te', hi_)))} |")
    return "\n".join(L)


def main():
    t0 = time.time()
    px, oh, val = load()
    merged, pm, base = collect(px, oh, val)
    nyears = (pd.Timestamp(px.index[-1]) - pd.Timestamp(px.index[0])).days / 365.25
    df = analyze(merged, base, nyears)
    df.to_csv(os.path.join(os.path.dirname(OUT), "strategies_all.csv"), index=False, encoding="utf-8-sig")
    md = report(df, pm, base, px)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(md)
    S.log(f"완료 {time.time() - t0:.0f}초")
    print(md)


if __name__ == "__main__":
    main()
