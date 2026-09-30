"""가상매매 연습(sim.html)용 시세 데이터를 만든다. sim_data.json 한 파일, 브라우저에서 fetch로 읽는다.
- 종목마다 종가는 그대로, 시/고/저는 종가와의 차이(정수)로 저장해 용량을 줄인다. (Pages가 gzip으로 전송)
- 거래대금은 백만원 단위 정수.
"""
import json

import numpy as np
import pandas as pd


def write_sim_data(px, oh, val, meta, path, max_stocks=600, min_days=700):
    dates = [str(d) for d in px.index]
    cnt = px.notna().sum()
    cand = [c for c in px.columns if cnt[c] >= min_days]
    tv = meta["거래대금"].reindex(cand).fillna(0)
    cand = list(tv.sort_values(ascending=False).index[:max_stocks])
    stocks = []
    for c in cand:
        cl = px[c]
        first = int(np.argmax(cl.notna().values))
        cl = cl.iloc[first:].ffill()
        o = oh["o"][c].iloc[first:].reindex(cl.index)
        h = oh["h"][c].iloc[first:].reindex(cl.index)
        l = oh["l"][c].iloc[first:].reindex(cl.index)
        v = val[c].iloc[first:].reindex(cl.index).fillna(0)
        gap = px[c].iloc[first:].isna().values          # 거래정지 등으로 빈 날: 직전 종가로 평평하게
        o = o.where(~gap, cl).fillna(cl)
        h = h.where(~gap, cl).fillna(cl)
        l = l.where(~gap, cl).fillna(cl)
        v = v.where(~gap, 0)
        ci = np.rint(cl.values).astype(np.int64)
        oi = np.rint(o.values).astype(np.int64)
        hi = np.maximum(np.rint(h.values).astype(np.int64), np.maximum(oi, ci))
        li = np.minimum(np.rint(l.values).astype(np.int64), np.minimum(oi, ci))
        stocks.append({
            "c": str(c), "n": str(meta.loc[c, "종목명"]), "m": str(meta.loc[c, "시장"]), "s": first,
            "cl": ci.tolist(), "o": (oi - ci).tolist(), "h": (hi - ci).tolist(), "l": (li - ci).tolist(),
            "v": np.rint(v.values / 1e6).astype(np.int64).tolist(),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dates": dates, "stocks": stocks}, f, ensure_ascii=False, separators=(",", ":"))
    return len(stocks)
