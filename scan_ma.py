"""
국내 전종목 이동평균선 근접 스캐너 (네이버 금융 시세 사용, 장 마감 후 실행용)

- 224 / 112 / 60일 이동평균선 대비 종가 이격이 ±GAP% 이내인 종목을 찾는다.
- 종목 목록과 일봉은 네이버 금융에서 받는다. (KRX/pykrx는 막혀서 사용하지 않음)
- 매 실행마다 전종목 일봉을 새로 받는다. 보통 3~8분 걸린다.
- 진행 상황은 scan_log.txt 에도 기록된다. 문제가 생기면 그 파일을 보면 된다.
- 결과는 results 폴더에 CSV(엑셀로 열림)로 저장한다.
"""
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests

# ---------------- 설정 (여기만 고치면 됨) ----------------
MA_LIST = [224, 112, 60]      # 볼 이동평균선
GAP_PCT = 10.0                # 이평선 대비 ±몇 % 이내를 '근접'으로 볼지
MIN_TRADING_VALUE = 1e9       # 그날 거래대금 하한(원). 1e9 = 10억
MIN_MARKET_CAP = 5e10         # 시가총액 하한(원). 5e10 = 500억
EXCLUDE_ETF_ETN = True        # ETF/ETN 제외
CHECK_RISK = True             # 관리종목/투자경고 등 위험종목 조회해서 제외
REQUIRE_ALL = False           # True면 세 이평선 모두 근접한 종목만, False면 하나라도 근접하면 포함
WORKERS = 6                   # 동시에 받는 개수 (너무 키우면 차단될 수 있음)
# ---------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "results")
LOG_PATH = os.path.join(BASE, "scan_log.txt")
COUNT = 1000                  # 종목당 받을 일봉 개수 (약 4년: 448일선과 주봉/월봉용)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

_log_lines = []
RAW_ITEMS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_lines.append(s)
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(_log_lines))
    except Exception:
        pass


def http_get(url, params=None, retries=3, timeout=15):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(1 + i)
    raise RuntimeError(f"요청 실패: {url} {params} -> {last}")


CODE_RE = re.compile(r'href="/item/main(?:\.naver)?\?code=(\d{6})"[^>]*>([^<]+)</a>')
JSON_CODE_RE = re.compile(r'"(?:itemCode|stockCode|code)"\s*:\s*"(\d{6})"')


def _list_via_api(market):
    """네이버 모바일 API로 시장별 전종목 코드/이름을 받는다."""
    names = {}
    page_size = 100
    for page in range(1, 60):
        r = http_get(
            f"https://m.stock.naver.com/api/stocks/marketValue/{market}",
            params={"page": page, "pageSize": page_size},
        )
        data = r.json()
        items = data.get("stocks") if isinstance(data, dict) else data
        if not items:
            if page == 1:
                log(f"API 응답 형태가 예상과 다름({market}):", str(data)[:600])
            break
        new = 0
        for it in items:
            RAW_ITEMS.append(dict(it, _market=market))
            code = str(it.get("itemCode") or it.get("stockCode") or it.get("code") or "")
            if re.fullmatch(r"\d{6}", code) and code not in names:
                names[code] = str(it.get("stockName") or it.get("name") or "").strip()
                new += 1
        if new == 0 or len(items) < page_size:
            break
        time.sleep(0.15)
    return names


def _list_via_html(sosok):
    """(예비) 옛 방식 HTML 목록. 네이버가 개편해서 실패할 수 있다."""
    names = {}
    for page in range(1, 80):
        r = http_get(
            "https://finance.naver.com/sise/sise_market_sum.naver",
            params={"sosok": sosok, "page": page},
        )
        html = r.content.decode("euc-kr", errors="replace")
        found = CODE_RE.findall(html)
        new = 0
        for code, name in found:
            if code not in names:
                names[code] = name.strip()
                new += 1
        if new == 0:
            break
        time.sleep(0.15)
    return names


def get_ticker_list():
    """코스피/코스닥 전종목 코드와 이름을 모은다."""
    names = {}
    for market, sosok in (("KOSPI", 0), ("KOSDAQ", 1)):
        got = {}
        try:
            got = _list_via_api(market)
        except Exception as e:
            log(f"{market} API 목록 실패:", repr(e)[:300])
        if len(got) < 500:
            log(f"{market} API로 {len(got)}개만 받음. 예비 방식 시도.")
            try:
                got2 = _list_via_html(sosok)
                if len(got2) > len(got):
                    got = got2
            except Exception as e:
                log(f"{market} 예비 방식도 실패:", repr(e)[:300])
        log(f"{market} 종목 수집: {len(got)}개")
        names.update(got)
    return names


ITEM_RE = re.compile(r'<item data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|(\d+)"')


def fetch_one(code):
    r = http_get(
        "https://fchart.stock.naver.com/sise.nhn",
        params={"symbol": code, "timeframe": "day", "count": COUNT, "requestType": 0},
    )
    rows = []
    for d, o, h, l, c, v in ITEM_RE.findall(r.text):
        close = float(c)
        vol = float(v)
        o_, h_, l_ = float(o), float(h), float(l)
        rows.append((d, code, o_, h_, l_, close, close * vol))  # 거래대금은 종가x거래량으로 근사
    return rows


def download_all(codes):
    rows = []
    fails = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for f in as_completed(futs):
            c = futs[f]
            try:
                rows.extend(f.result())
            except Exception as e:
                fails.append((c, str(e)[:120]))
            done += 1
            if done % 200 == 0 or done == len(codes):
                el = time.time() - t0
                log(f"  시세 {done}/{len(codes)} 완료 (실패 {len(fails)}) {el:.0f}초")
            # 초반에 전부 실패하면 오래 기다리지 말고 바로 멈춘다
            if done == 30 and len(fails) >= 25:
                log("초반 30개 중 25개 이상 실패. 접속 문제로 판단하고 중단.")
                log("실패 예시:", fails[:3])
                ex.shutdown(wait=False, cancel_futures=True)
                break
    return rows, fails


RISK_WORDS = ["관리종목", "투자경고", "투자위험", "거래정지", "정리매매", "상장폐지", "불성실공시", "단기과열"]


def fetch_risk(code):
    """종목별 기본정보에서 위험 지정 문구를 찾는다. 반환: (문구목록 또는 None=조회실패, 원본json)"""
    try:
        r = http_get(f"https://m.stock.naver.com/api/stock/{code}/basic", retries=2)
        data = r.json()
        text = json.dumps(data, ensure_ascii=False)
        return [w for w in RISK_WORDS if w in text], data
    except Exception as e:
        return None, {"error": repr(e)[:200]}


def check_risk(codes):
    flags, samples, fail = {}, [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_risk, c): c for c in codes}
        for f in as_completed(futs):
            c = futs[f]
            fl, raw = f.result()
            if fl is None:
                fail += 1
            else:
                flags[c] = fl
            if len(samples) < 3:
                samples.append({"code": c, "flags": fl, "raw": raw})
    try:
        with open(os.path.join(BASE, "risk_sample.json"), "w", encoding="utf-8") as fh:
            json.dump(samples, fh, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return flags, fail


def get_px(cache):
    px = cache.pivot(index="날짜", columns="티커", values="종가").sort_index()
    px = px.where(px > 0)
    cnt = px.notna().sum(axis=1)
    return px[cnt >= cnt.max() * 0.5]


def get_ohlc(cache, px):
    """px(종가 피벗)와 같은 모양의 시가/고가/저가 피벗. 0원(거래정지일 등)은 종가로 대체."""
    out = {}
    for key, col in (("o", "시가"), ("h", "고가"), ("l", "저가")):
        t = cache.pivot(index="날짜", columns="티커", values=col).reindex(index=px.index, columns=px.columns)
        out[key] = t.where(t > 0, px)
    return out


def scan(cache, meta, ma_list=MA_LIST, gap_pct=GAP_PCT):
    """이평선별로 근접 종목을 계산한다. (네트워크 없이 동작)
    meta: 종목코드 인덱스, 컬럼 [종목명, 시장, 구분, 시총, 거래대금, 거래상태]"""
    px = cache.pivot(index="날짜", columns="티커", values="종가").sort_index()
    px = px.where(px > 0)
    cnt = px.notna().sum(axis=1)
    good = cnt >= cnt.max() * 0.5  # 휴장일 등 데이터가 비정상적으로 적은 날 제외
    px = px[good]
    last_day = px.index[-1]
    last_close = px.iloc[-1]

    m = meta.reindex(px.columns)
    ok = last_close.notna()
    if EXCLUDE_ETF_ETN:
        ok &= m["구분"].eq("stock")
    ok &= m["거래상태"].eq("tradable")
    ok &= m["시총"].fillna(0) >= MIN_MARKET_CAP
    ok &= m["거래대금"].fillna(0) >= MIN_TRADING_VALUE

    gaps = pd.DataFrame({n: (last_close / px.rolling(n, min_periods=n).mean().iloc[-1] - 1) * 100
                         for n in ma_list})
    base = pd.DataFrame({
        "종목명": m["종목명"], "시장": m["시장"], "종가": last_close,
        "시총(억)": (m["시총"] / 1e8).round(0), "거래대금(억)": (m["거래대금"] / 1e8).round(1),
    })
    results = {}
    for n in ma_list:
        sel = ok & gaps[n].abs().le(gap_pct)
        df = base[sel].copy()
        df[f"{n}일선 이격(%)"] = gaps.loc[sel, n]
        for k in ma_list:
            if k != n:
                df[f"{k}일선 이격(%)"] = gaps.loc[sel, k]
        df = df.reindex(sorted(df.index, key=lambda t: abs(gaps.loc[t, n])))
        results[n] = df.round(2)
    return last_day, results


def load_meta():
    raw = pd.DataFrame(RAW_ITEMS)
    meta = pd.DataFrame({
        "종목명": raw["stockName"], "시장": raw["_market"], "구분": raw["stockEndType"],
        "시총": pd.to_numeric(raw["marketValueRaw"], errors="coerce"),
        "거래대금": pd.to_numeric(raw["accumulatedTradingValueRaw"], errors="coerce"),
        "거래상태": raw["tradableStatus"],
    })
    meta.index = raw["itemCode"].astype(str)
    return meta[~meta.index.duplicated()]


def save_results(results, last_day):
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import openpyxl  # noqa: F401
        path = os.path.join(OUT_DIR, f"ma_scan_{last_day}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for n, df in results.items():
                d = df.copy()
                d.index.name = "티커"
                d["토스"] = [f'=HYPERLINK("https://www.tossinvest.com/stocks/A{c}","열기")' for c in d.index]
                sheet = f"{n}일선"
                d.to_excel(w, sheet_name=sheet)
                ws = w.sheets[sheet]
                ws.freeze_panes = "C2"
                for col, width in zip("ABCDEFGHIJK", (9, 18, 9, 10, 10, 12, 14, 14, 14, 8, 8)):
                    ws.column_dimensions[col].width = width
        return [path]
    except ImportError:
        paths = []
        for n, df in results.items():
            d = df.copy()
            d.index.name = "티커"
            d["토스"] = [f"https://www.tossinvest.com/stocks/A{c}" for c in d.index]
            p = os.path.join(OUT_DIR, f"ma_{n}_{last_day}.csv")
            d.to_csv(p, encoding="utf-8-sig")
            paths.append(p)
        log("(openpyxl이 없어서 CSV 3개로 저장. 엑셀 한 파일로 받으려면: pip install openpyxl)")
        return paths


def main():
    log("시작", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        names = get_ticker_list()
        if len(names) < 1000:
            log(f"종목 수가 너무 적음({len(names)}개). 목록 수집 실패로 중단.")
            sys.exit(1)
        try:
            pd.json_normalize(RAW_ITEMS).to_csv(os.path.join(BASE, "stocks_raw.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:
            log("stocks_raw 저장 실패:", repr(e)[:200])
        meta = load_meta()
        log(f"전체 {len(meta)}종목 (ETF {int((meta['구분'] == 'etf').sum())}, ETN {int((meta['구분'] == 'etn').sum())}). 시세 받는 중...")

        # 필터를 통과할 수 있는 종목만 시세를 받는다 (속도 절약)
        want = meta.index[
            (meta["구분"].eq("stock") if EXCLUDE_ETF_ETN else True)
            & meta["거래상태"].eq("tradable")
            & (meta["시총"].fillna(0) >= MIN_MARKET_CAP)
            & (meta["거래대금"].fillna(0) >= MIN_TRADING_VALUE)
        ].tolist()
        log(f"시총/거래대금/ETF 조건 통과: {len(want)}종목만 시세 조회")
        rows, fails = download_all(want)
        if not rows:
            log("시세를 하나도 못 받았어.")
            sys.exit(1)
        cache = pd.DataFrame(rows, columns=["날짜", "티커", "시가", "고가", "저가", "종가", "거래대금"])
        log(f"수신 완료: {cache['티커'].nunique()}종목, 실패 {len(fails)}개")
        if fails:
            log("실패 예시:", fails[:3])

        last_day, results = scan(cache, meta)

        if CHECK_RISK:
            cand = sorted(set().union(*[set(df.index) for df in results.values()]))
            log(f"위험종목 조회 대상(후보 합계): {len(cand)}종목")
            flags, rfail = check_risk(cand)
            bad = {c: f for c, f in flags.items() if f}
            log(f"위험 문구 발견 {len(bad)}종목, 조회 실패 {rfail}종목")
            if bad:
                log("위험 예시:", [(meta['종목명'].get(c), f) for c, f in list(bad.items())[:5]])
            if rfail > len(cand) * 0.5:
                log("조회 실패가 많아서 위험종목 필터는 적용하지 않음(risk_sample.json 확인 필요).")
                bad = {}
            for n, df in results.items():
                results[n] = df[~df.index.isin(bad)]

        paths = save_results(results, last_day)
        try:
            from report import build_report
            html_path = os.path.join(OUT_DIR, f"ma_scan_{last_day}.html")
            px_all = get_px(cache)
            build_report(px_all, get_ohlc(cache, px_all), results, meta, last_day, MA_LIST, GAP_PCT, html_path)
            import shutil
            shutil.copyfile(html_path, os.path.join(OUT_DIR, "latest.html"))
            paths.append(html_path)
        except Exception:
            log("차트 페이지 생성 실패(엑셀은 정상 저장됨):")
            log(traceback.format_exc()[-800:])
        log(f"\n기준일 {last_day}")
        for n, df in results.items():
            log(f"  {n}일선 ±{GAP_PCT:g}% 이내: {len(df)}종목")
        log("저장:", *paths)
        log("끝")
    except SystemExit:
        raise
    except Exception:
        log("에러 발생:")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
