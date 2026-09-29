"""스캔 결과를 종목별 캔들차트 카드 HTML 한 장으로 만든다. (인터넷 없이도 열림, 외부 라이브러리 없음)
카드/확대 화면의 그림과 이평선 계산은 전부 브라우저(JS)에서 한다. 파이썬은 일봉 OHLC만 넘긴다."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

KDAYS = 1000   # 브라우저로 넘길 일봉 개수 (약 4년)
# 화면에 그리는 이평선과 색 (스캔 기준 이평선 60/112/224와는 별개)
SHOW_MAS = [5, 20, 60, 112, 224, 448]
SHOW_COLORS = {5: "#ff3b30", 20: "#facc15", 60: "#fb923c", 112: "#22c55e", 224: "#ffffff", 448: "#3b82f6"}


def _num(v):
    if pd.isna(v):
        return None
    x = round(float(v), 2)
    return int(x) if x == int(x) else x


def build_report(px, ohlc, results, meta, last_day, ma_list, gap_pct, path):
    cand = sorted(set().union(*[set(df.index) for df in results.values()]))
    mas = {k: px.rolling(k, min_periods=k).mean() for k in ma_list}
    close = px.iloc[-1]

    stocks = {}
    for c in cand:
        gaps = {k: (round(float(close[c] / mas[k][c].iloc[-1] * 100 - 100), 2)
                    if pd.notna(mas[k][c].iloc[-1]) else None) for k in ma_list}
        stocks[c] = {
            "n": str(meta.loc[c, "종목명"]), "m": str(meta.loc[c, "시장"]),
            "p": float(close[c]), "cap": float(meta.loc[c, "시총"]) / 1e8,
            "val": float(meta.loc[c, "거래대금"]) / 1e8, "g": gaps,
        }
    tabs = {str(k): list(df.index) for k, df in results.items()}

    kidx = px.index[-KDAYS:]
    kdata = {}
    for c in cand:
        kdata[c] = {
            "o": [_num(v) for v in ohlc["o"][c].reindex(kidx)],
            "h": [_num(v) for v in ohlc["h"][c].reindex(kidx)],
            "l": [_num(v) for v in ohlc["l"][c].reindex(kidx)],
            "c": [_num(v) for v in px[c].reindex(kidx)],
        }

    gen = datetime.now(timezone(timedelta(hours=9))).strftime("%m-%d %H:%M")
    payload = json.dumps(
        {"day": str(last_day), "gen": gen, "gap": gap_pct, "mas": ma_list, "dm": SHOW_MAS,
         "mc": {str(k): v for k, v in SHOW_COLORS.items()},
         "tabs": tabs, "stocks": stocks, "dates": list(kidx), "k": kdata},
        ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__DATA__", payload))
    return path


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>이평선 근접 종목</title>
<style>
:root{color-scheme:dark;--bg:#000;--card:#0d0f12;--tx:#e8eaed;--sub:#8b929c;--line:#23272e;--acc:#4c8dff;--up:#ff4d4d;--dn:#4c8dff}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
@media (min-width:1001px){html{scroll-snap-type:y proximity}}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.4 system-ui,-apple-system,"Malgun Gothic",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center}
h1{font-size:16px;margin:0 6px 0 0;white-space:nowrap}
.sub{color:var(--sub);font-size:12px;font-weight:400}
.tab{border:1px solid var(--line);background:var(--card);color:var(--tx);padding:5px 11px;border-radius:8px;cursor:pointer;font:inherit}
.tab[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
select,label.chk{border:1px solid var(--line);background:var(--card);color:var(--tx);padding:4px 8px;border-radius:8px;font:inherit}
label.chk{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.legend{display:flex;gap:11px;font-size:12px;color:var(--sub);margin-left:auto;align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
.legend b{display:inline-block;width:8px;height:12px;vertical-align:middle;margin-right:4px;border-radius:1px}
main{--rowh:280px;padding:10px 14px 14px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-auto-rows:var(--rowh);gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px 12px 8px;display:flex;flex-direction:column;min-height:0;scroll-snap-align:start;scroll-margin-top:var(--hh,90px);cursor:pointer}
.card{-webkit-tap-highlight-color:transparent}
.card:hover{border-color:var(--acc)}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.nm{font-weight:700;font-size:16px;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk{font-size:12px;color:var(--sub);white-space:nowrap}
a.tl{color:var(--acc);text-decoration:none;margin-left:6px}
a.tl:hover{text-decoration:underline}
.badges{display:flex;gap:6px;margin:5px 0 4px;flex-wrap:wrap}
.b{font-size:12px;padding:1px 7px;border-radius:6px;border:1px solid var(--line);color:var(--sub)}
.b.f{color:var(--tx);font-weight:700;border-width:2px}
.up{color:var(--up)}.dn{color:var(--dn)}
.chart{flex:1;min-height:0}
svg{width:100%;height:100%;display:block}
svg .wu,svg .wd{fill:none;stroke-width:1;vector-effect:non-scaling-stroke}
svg .wu{stroke:var(--up)}svg .wd{stroke:var(--dn)}
svg .bu,svg .bd{fill:none;stroke-linecap:butt}
svg .bu{stroke:var(--up)}svg .bd{stroke:var(--dn)}
svg .m{fill:none;opacity:.9;vector-effect:non-scaling-stroke}
.ft{display:flex;justify-content:space-between;color:var(--sub);font-size:12px;margin-top:3px;white-space:nowrap;gap:6px}
.empty{grid-column:1/-1;color:var(--sub);padding:40px;text-align:center}
#ov{position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.72);display:none;align-items:center;justify-content:center}
#ov.on{display:flex}
#pn{background:var(--card);border:1px solid var(--line);border-radius:14px;width:min(97vw,1800px);height:95vh;height:95dvh;display:flex;flex-direction:column;padding:10px 14px 12px;box-shadow:0 10px 40px rgba(0,0,0,.6)}
#ph{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
#pt{font-size:19px;font-weight:700}
#pb{display:flex;gap:6px;flex-wrap:wrap}
#ptool{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0;padding:6px 8px;background:#15181d;border-radius:10px}
.tb{border:1px solid var(--line);background:var(--card);color:var(--tx);padding:5px 11px;border-radius:8px;cursor:pointer;font:inherit}
.tb[aria-pressed=true]{background:var(--acc);border-color:var(--acc);color:#fff}
.tb:hover{border-color:var(--acc)}
.tf{display:inline-flex;gap:0;margin:0 6px}
.tf .tb{border-radius:0}.tf .tb:first-child{border-radius:8px 0 0 8px}.tf .tb:last-child{border-radius:0 8px 8px 0}
.sw{width:22px;height:22px;border-radius:50%;border:2px solid var(--line);cursor:pointer;padding:0}
.sw[aria-pressed=true]{border-color:#fff}
.sep{width:1px;height:22px;background:var(--line)}
.hint{color:var(--sub);font-size:12px;margin-left:auto}
#cw{flex:1;min-height:0;position:relative}
#cv{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;display:block;touch-action:none;-webkit-user-select:none;user-select:none}
.x{margin-left:auto;font-size:22px;line-height:1;padding:2px 10px}
.pg{font-size:12px;color:var(--sub)}
.hint.mo{display:none}
@media (max-width:1000px){main{grid-template-columns:repeat(3,minmax(0,1fr));--rowh:270px !important}}
@media (max-width:700px){
  main{grid-template-columns:repeat(2,minmax(0,1fr));--rowh:250px !important;padding:8px;gap:8px}
  header{position:static;padding:8px}
  h1{white-space:normal;font-size:15px;margin:0;flex:1 0 100%}
  h1 .sub{display:block}
  .card{padding:7px 8px 6px;scroll-margin-top:0}
  .top{flex-direction:column;align-items:flex-start;gap:0}
  .nm{font-size:14px;max-width:100%}.mk{font-size:11px}
  .b{font-size:11px;padding:0 5px}
  .badges{gap:4px;margin:3px 0}
  .ft{flex-direction:column;gap:0;font-size:11px;white-space:normal}
  .legend{margin-left:0}
  #pn{width:100vw;height:100vh;height:100dvh;border-radius:0;padding:8px 8px 10px}
  #pt{font-size:16px}
}
@media (pointer:coarse){
  .tb,.tab,select,label.chk{padding:8px 12px}
  .sw{width:30px;height:30px}
  .hint{display:none}.hint.mo{display:block}
}
</style></head><body>
<header id="hd">
  <h1>이평선 근접 종목 <span class="sub" id="day"></span></h1>
  <span id="tabs"></span>
  <select id="sort"><option value="near">이평선에 가까운 순</option><option value="cap">시가총액 큰 순</option><option value="val">거래대금 큰 순</option></select>
  <select id="mk"><option value="">코스피+코스닥</option><option value="KOSPI">코스피</option><option value="KOSDAQ">코스닥</option></select>
  <select id="pos"><option value="">위치 전체</option><option value="above">이평선 위 (눌림)</option><option value="below">이평선 아래 (이탈)</option></select>
  <label class="chk"><input type="checkbox" id="all3"> 세 이평선 모두 근접</label>
  <span class="sub" id="cnt"></span>
  <div class="legend" id="legend"></div>
</header>
<main id="grid"></main>
<div id="ov"><div id="pn">
  <div id="ph"><span id="pt"></span><span class="mk" id="pm"></span><span id="pb"></span>
    <span class="tf" id="ptf"><button class="tb" data-tf="D" aria-pressed="true">일봉</button><button class="tb" data-tf="W" aria-pressed="false">주봉</button><button class="tb" data-tf="M" aria-pressed="false">월봉</button></span>
    <a class="tb" id="ptoss" target="_blank" rel="noopener" style="text-decoration:none">토스증권 ↗</a><span class="pg" id="ppg"></span>
    <button class="tb" id="pprev" title="이전 종목 (←)">◀</button><button class="tb" id="pnext" title="다음 종목 (→)">▶</button><button class="tb x" id="pclose" title="닫기 (Esc)">×</button></div>
  <div id="ptool">
    <button class="tb" data-tool="cursor" aria-pressed="true">↖ 이동/선택</button>
    <button class="tb" data-tool="trend" aria-pressed="false">╱ 추세선</button>
    <button class="tb" data-tool="h" aria-pressed="false">─ 수평선</button>
    <span class="sep"></span><span id="sws"></span><span class="sep"></span>
    <label class="chk"><input type="checkbox" id="pext" checked> 오른쪽 연장</label>
    <label class="chk"><input type="checkbox" id="pmag" checked> 자석(고/저/시/종)</label>
    <span class="sep"></span>
    <button class="tb" id="pundo">되돌리기</button><button class="tb" id="pdel">선택 삭제</button><button class="tb" id="pclr">전체 삭제</button>
    <span class="hint">휠: 확대/축소 · 드래그: 이동 · 더블클릭: 처음 보기 · Del: 선택 삭제 · Esc: 닫기</span><span class="hint mo">탭: 고가/저가 · 두 손가락: 확대/축소 · 드래그: 이동 · 일/주/월 버튼 다시 탭: 처음 보기</span>
  </div>
  <div id="cw"><canvas id="cv"></canvas></div>
</div></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const DM=D.dm,MC=D.mc,DATES=D.dates,NBD=DATES.length,KD=D.k;
let cur=String(D.mas[D.mas.length-1]);
$("#day").textContent="· 기준일 "+D.day.replace(/(\d{4})(\d\d)(\d\d)/,"$1-$2-$3")+" · ±"+D.gap+"% 이내"+(D.gen?" · 갱신 "+D.gen:"");
$("#legend").innerHTML='<span><b style="background:var(--up)"></b>양봉</span><span><b style="background:var(--dn)"></b>음봉</span>'+DM.map(k=>`<span><i style="background:${MC[k]}"></i>${k}일</span>`).join("");
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(g){if(g===null)return "-";const r=Math.round(g*10)/10;if(r===0)return "0.0%";return (r>0?"+":"")+r.toFixed(1)+"%"}
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

/* ---------- 데이터: 일/주/월 봉과 이평선 (전부 여기서 계산) ---------- */
function sma(c,p){const n=c.length,o=new Array(n).fill(null);let s=0,cnt=0;
  for(let i=0;i<n;i++){if(c[i]!=null){s+=c[i];cnt++}if(i>=p&&c[i-p]!=null){s-=c[i-p];cnt--}if(i>=p-1&&cnt===p)o[i]=s/p}return o}
const DC={};
function daily(code){if(DC[code])return DC[code];const k=KD[code];
  const S={n:NBD,d:DATES,o:k.o,h:k.h,l:k.l,c:k.c,m:{},map:new Map(DATES.map((d,i)=>[d,i]))};
  DM.forEach(p=>S.m[p]=sma(k.c,p));return DC[code]=S}
function agg(code,tf){const k=KD[code];
  const keyOf=d=>{if(tf==="M")return d.slice(0,6);const t=new Date(Date.UTC(+d.slice(0,4),+d.slice(4,6)-1,+d.slice(6)));t.setUTCDate(t.getUTCDate()-((t.getUTCDay()+6)%7));return t.getTime()};
  const S={d:[],o:[],h:[],l:[],c:[],m:{},map:new Map()};let last=null,bi=-1;
  for(let i=0;i<NBD;i++){if(k.c[i]==null)continue;const key=keyOf(DATES[i]);
    if(key!==last){bi++;last=key;S.d.push(DATES[i]);S.o.push(k.o[i]);S.h.push(k.h[i]);S.l.push(k.l[i]);S.c.push(k.c[i])}
    else{S.d[bi]=DATES[i];S.h[bi]=Math.max(S.h[bi],k.h[i]);S.l[bi]=Math.min(S.l[bi],k.l[i]);S.c[bi]=k.c[i]}
    S.map.set(DATES[i],bi)}
  S.n=S.d.length;DM.forEach(p=>S.m[p]=sma(S.c,p));return S}

/* ---------- 카드 (일봉 최근 120개 + 이평선 6개) ---------- */
function cardSvg(code,focus){
  const S=daily(code),VW=480,VH=230,DAYS=120,a=Math.max(0,S.n-DAYS);
  let lo=Infinity,hi=-Infinity;
  for(let i=a;i<S.n;i++)if(S.h[i]!=null){hi=Math.max(hi,S.h[i]);lo=Math.min(lo,S.l[i])}
  if(!isFinite(lo))return"";
  const sp=hi-lo||1;let l2=lo,h2=hi;
  DM.forEach(p=>{for(let i=a;i<S.n;i++){const v=S.m[p][i];if(v!=null&&v>=lo-.5*sp&&v<=hi+.5*sp){l2=Math.min(l2,v);h2=Math.max(h2,v)}}});
  const pad=(h2-l2)*.04||1;l2-=pad;h2+=pad;
  const Y=v=>(VH-6-(v-l2)/(h2-l2)*(VH-12)).toFixed(1),pitch=VW/DAYS,X=i=>((i-S.n+DAYS)+.5)*pitch;
  let wu="",wd="",bu="",bd="",ls="";
  for(let i=a;i<S.n;i++){if(S.c[i]==null)continue;const x=X(i).toFixed(1),o=S.o[i],c=S.c[i];let y1=+Y(o),y2=+Y(c);
    if(Math.abs(y1-y2)<.8){const m=(y1+y2)/2;y1=m+.4;y2=m-.4}
    const w=`M${x} ${Y(S.h[i])}V${Y(S.l[i])}`,b=`M${x} ${y1}V${y2}`;
    if(c>=o){wu+=w;bu+=b}else{wd+=w;bd+=b}}
  DM.slice().reverse().forEach(p=>{let pts="";for(let i=a;i<S.n;i++){const v=S.m[p][i];if(v!=null)pts+=X(i).toFixed(1)+","+Y(v)+" "}
    if(pts){const f=String(p)===focus;ls+=`<polyline class="m" points="${pts}" style="stroke:${MC[p]};stroke-width:${f?2.6:(p===5?.9:1.3)}"/>`}});
  const bw=(pitch*.68).toFixed(2);
  return `<svg viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="none" aria-hidden="true">${ls}<path class="wu" d="${wu}"/><path class="wd" d="${wd}"/><path class="bu" style="stroke-width:${bw}" d="${bu}"/><path class="bd" style="stroke-width:${bw}" d="${bd}"/></svg>`;
}

function layout(){
  const hd=$("#hd").offsetHeight;document.documentElement.style.setProperty("--hh",hd+"px");
  if(innerWidth>1000){const avail=innerHeight-hd-24-10*2;$("#grid").style.setProperty("--rowh",Math.max(230,Math.floor(avail/3))+"px")}
}
function tabs(){
  $("#tabs").innerHTML=[...D.mas].sort((a,b)=>b-a).map(k=>`<button class="tab" data-k="${k}" aria-pressed="${String(k)===cur}">${k}일선 (${D.tabs[k].length})</button>`).join(" ");
  document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{cur=b.dataset.k;tabs();render();scrollTo(0,0)});
}
function render(){
  let list=D.tabs[cur].map(c=>({c,s:D.stocks[c]}));
  const mk=$("#mk").value,pos=$("#pos").value,all3=$("#all3").checked,so=$("#sort").value;
  list=list.filter(({s})=>(!mk||s.m===mk)&&(!pos||(pos==="above"?s.g[cur]>=0:s.g[cur]<0))&&(!all3||D.mas.every(k=>s.g[k]!==null&&Math.abs(s.g[k])<=D.gap)));
  if(so==="cap")list.sort((a,b)=>b.s.cap-a.s.cap);else if(so==="val")list.sort((a,b)=>b.s.val-a.s.val);
  else list.sort((a,b)=>Math.abs(a.s.g[cur])-Math.abs(b.s.g[cur]));
  $("#cnt").textContent=list.length+"종목";window.CUR=list.map(x=>x.c);
  $("#grid").innerHTML=list.length?list.map(({c,s})=>{
    const bd=[...D.mas].sort((a,b)=>b-a).map(k=>`<span class="b ${String(k)===cur?"f":""} ${s.g[k]>0?"up":"dn"}">${k}일 ${fmt(s.g[k])}</span>`).join("");
    return `<div class="card" data-c="${c}"><div class="top"><span class="nm">${esc(s.n)}</span><span class="mk">${c} · ${s.m==="KOSPI"?"코스피":"코스닥"}<a class="tl" target="_blank" rel="noopener" href="https://www.tossinvest.com/stocks/A${c}" title="토스증권에서 열기">토스↗</a></span></div>
    <div class="badges">${bd}</div><div class="chart">${cardSvg(c,cur)}</div>
    <div class="ft"><span>종가 ${Math.round(s.p).toLocaleString()}원</span><span>시총 ${Math.round(s.cap).toLocaleString()}억 · 거래 ${s.val.toFixed(0)}억</span></div></div>`}).join(""):'<div class="empty">조건에 맞는 종목이 없어</div>';
}
["#sort","#mk","#pos","#all3"].forEach(s=>$(s).onchange=()=>{render();scrollTo(0,0)});
addEventListener("resize",layout);

/* ---------- 확대 화면: 일/주/월봉 + 그리기 도구 ---------- */
const SWC=["#ec4899","#06b6d4","#84cc16","#a855f7","#94a3b8"];/* 이평선/캔들 색과 안 겹치는 색 */
let M=null,G=null,swColor=SWC[0];
const cv=$("#cv"),cx=cv.getContext("2d");
$("#sws").innerHTML=SWC.map((c,i)=>`<button class="sw" data-c="${c}" style="background:${c}" aria-pressed="${i===0}"></button>`).join(" ");
document.querySelectorAll(".sw").forEach(b=>b.onclick=()=>{swColor=b.dataset.c;document.querySelectorAll(".sw").forEach(x=>x.setAttribute("aria-pressed",x===b));
  if(M&&M.sel>=0){M.lines[M.sel].col=swColor;save();draw()}});
function geo(){const W=cv.clientWidth,H=cv.clientHeight,t=W<1100?58:38,r=W<700?66:74;return{W,H,l:8,r,t,b:26,pw:W-8-r,ph:H-t-26}}
const xOf=i=>G.l+(i-M.start+.5)*G.pw/M.N, iOf=x=>Math.round((x-G.l)/(G.pw/M.N)-.5+M.start);
const yOf=p=>G.t+(M.hi-p)/(M.hi-M.lo)*G.ph, pOf=y=>M.hi-(y-G.t)/G.ph*(M.hi-M.lo);
const fmtP=p=>Math.round(p).toLocaleString();
function range(){
  const S=M.S,a=Math.max(0,Math.floor(M.start)),b=Math.min(S.n-1,Math.ceil(M.start+M.N));let lo=Infinity,hi=-Infinity;
  for(let i=a;i<=b;i++)if(S.h[i]!=null){hi=Math.max(hi,S.h[i]);lo=Math.min(lo,S.l[i])}
  if(!isFinite(lo)){lo=0;hi=1}
  const sp=hi-lo||1;let l2=lo,h2=hi;
  DM.forEach(k=>{for(let i=a;i<=b;i++){const v=S.m[k][i];if(v!=null&&v>=lo-.5*sp&&v<=hi+.5*sp){l2=Math.min(l2,v);h2=Math.max(h2,v)}}});
  const pad=(h2-l2)*.06||1;M.lo=l2-pad;M.hi=h2+pad;
}
function niceStep(r){const raw=r/6,p=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/p;return(f<1.5?1:f<3?2:f<7?5:10)*p}
function lineXY(L){
  if(L.t==="h")return{x1:G.l,y1:yOf(L.p),x2:G.l+G.pw,y2:yOf(L.p)};
  const i1=M.S.map.get(L.d1),i2=M.S.map.get(L.d2);if(i1==null||i2==null||i1===i2)return null;
  let x1=xOf(i1),y1=yOf(L.p1),x2=xOf(i2),y2=yOf(L.p2);
  if(L.ext){const sl=(y2-y1)/(x2-x1),xe=G.l+G.pw;if(x2<xe){y2=y2+sl*(xe-x2);x2=xe}}
  return{x1,y1,x2,y2};
}
function pill(text,x,y,col){ // 글자 상자 (x: 가운데, y: 위쪽)
  cx.font="12px system-ui,'Malgun Gothic',sans-serif";const w=cx.measureText(text).width+12,h=18;
  let px=Math.min(Math.max(x-w/2,G.l+2),G.l+G.pw-w-2),py=Math.min(Math.max(y,G.t+2),G.t+G.ph-h-2);
  cx.fillStyle="#000c";cx.strokeStyle=col;cx.lineWidth=1;cx.beginPath();cx.roundRect(px,py,w,h,4);cx.fill();cx.stroke();
  cx.fillStyle=col;cx.textAlign="left";cx.textBaseline="middle";cx.fillText(text,px+6,py+h/2+.5);
}
function draw(){
  if(!M)return;
  const dpr=devicePixelRatio||1,cw=$("#cw");
  cv.width=cw.clientWidth*dpr;cv.height=cw.clientHeight*dpr;cx.setTransform(dpr,0,0,dpr,0,0);
  G=geo();range();const S=M.S,tx=cssv("--tx"),sub=cssv("--sub"),ln=cssv("--line"),up=cssv("--up"),dn=cssv("--dn");
  cx.font="12px system-ui,'Malgun Gothic',sans-serif";cx.clearRect(0,0,G.W,G.H);
  const st=niceStep(M.hi-M.lo);cx.textAlign="left";cx.textBaseline="middle";
  for(let p=Math.ceil(M.lo/st)*st;p<=M.hi;p+=st){const y=yOf(p);cx.strokeStyle=ln;cx.lineWidth=1;cx.beginPath();cx.moveTo(G.l,y);cx.lineTo(G.l+G.pw,y);cx.stroke();cx.fillStyle=sub;cx.fillText(fmtP(p),G.l+G.pw+6,y)}
  // 날짜축: 일/주봉은 월 바뀔 때, 월봉은 해 바뀔 때
  cx.textAlign="center";cx.textBaseline="top";let pm="",lx=-99;const a=Math.max(0,Math.floor(M.start)),b=Math.min(S.n-1,Math.ceil(M.start+M.N));
  for(let i=a;i<=b;i++){const d=S.d[i],key=M.tf==="M"?d.slice(0,4):d.slice(0,6);
    if(key!==pm){pm=key;const x=xOf(i);if(x>G.l&&x<G.l+G.pw-24&&x-lx>=36){lx=x;cx.strokeStyle=ln;cx.beginPath();cx.moveTo(x,G.t);cx.lineTo(x,G.t+G.ph);cx.stroke();cx.fillStyle=sub;
      cx.fillText(M.tf==="M"?d.slice(0,4)+"년":(d.slice(4,6)==="01"?d.slice(0,4)+"년":(+d.slice(4,6))+"월"),x,G.t+G.ph+6)}}}
  cx.save();cx.beginPath();cx.rect(G.l,G.t,G.pw,G.ph);cx.clip();
  DM.slice().reverse().forEach(k=>{cx.strokeStyle=MC[k];cx.lineWidth=k===5?1:1.7;cx.beginPath();let on=false;
    for(let i=0;i<S.n;i++){const v=S.m[k][i];if(v==null){on=false;continue}const x=xOf(i),y=yOf(v);if(!on){cx.moveTo(x,y);on=true}else cx.lineTo(x,y)}cx.stroke()});
  const bw=G.pw/M.N,w=Math.max(1,bw*.66);
  for(let i=a;i<=b;i++){if(S.c[i]==null)continue;const o=S.o[i],c=S.c[i],col=c>=o?up:dn,x=xOf(i);
    cx.strokeStyle=col;cx.fillStyle=col;cx.lineWidth=1;cx.beginPath();cx.moveTo(x,yOf(S.h[i]));cx.lineTo(x,yOf(S.l[i]));cx.stroke();
    const y1=yOf(Math.max(o,c)),y2=yOf(Math.min(o,c));cx.fillRect(x-w/2,y1,w,Math.max(1,y2-y1))}
  M.lines.forEach((L,idx)=>{const s=lineXY(L);if(!s)return;const sel=idx===M.sel;
    cx.strokeStyle=L.col;cx.lineWidth=sel?3.2:1.8;cx.beginPath();cx.moveTo(s.x1,s.y1);cx.lineTo(s.x2,s.y2);cx.stroke();
    if(sel&&L.t==="trend"){[[xOf(M.S.map.get(L.d1)),yOf(L.p1)],[xOf(M.S.map.get(L.d2)),yOf(L.p2)]].forEach(([x,y])=>{cx.fillStyle=L.col;cx.beginPath();cx.arc(x,y,4.5,0,7);cx.fill()})}});
  if(M.tmp&&M.mouse){const s=snap(M.mouse.x,M.mouse.y);cx.strokeStyle=swColor;cx.setLineDash([6,4]);cx.lineWidth=1.6;cx.beginPath();cx.moveTo(xOf(M.tmp.i),yOf(M.tmp.p));cx.lineTo(xOf(s.i),yOf(s.p));cx.stroke();cx.setLineDash([]);
    cx.fillStyle=swColor;cx.beginPath();cx.arc(xOf(M.tmp.i),yOf(M.tmp.p),4,0,7);cx.fill()}
  cx.restore();
  M.lines.forEach(L=>{if(L.t!=="h")return;const y=yOf(L.p);if(y<G.t||y>G.t+G.ph)return;cx.fillStyle=L.col;cx.fillRect(G.l+G.pw+1,y-9,G.r-4,18);cx.fillStyle="#000";cx.textAlign="left";cx.textBaseline="middle";cx.fillText(fmtP(L.p),G.l+G.pw+5,y)});
  let hi_i=S.n-1;
  const inside=M.mouse&&M.mouse.x>=G.l&&M.mouse.x<=G.l+G.pw&&M.mouse.y>=G.t&&M.mouse.y<=G.t+G.ph;
  if(inside){
    const i=Math.min(S.n-1,Math.max(0,iOf(M.mouse.x)));hi_i=i;const x=xOf(i);
    cx.strokeStyle=sub;cx.setLineDash([3,4]);cx.lineWidth=1;cx.beginPath();cx.moveTo(x,G.t);cx.lineTo(x,G.t+G.ph);cx.moveTo(G.l,M.mouse.y);cx.lineTo(G.l+G.pw,M.mouse.y);cx.stroke();cx.setLineDash([]);
    const p=pOf(M.mouse.y);cx.fillStyle=tx;cx.fillRect(G.l+G.pw+1,M.mouse.y-9,G.r-4,18);cx.fillStyle="#000";cx.textAlign="left";cx.textBaseline="middle";cx.fillText(fmtP(p),G.l+G.pw+5,M.mouse.y);
    // 마우스가 올라간 봉의 고가/저가 표시
    if(S.c[i]!=null){const yh=yOf(S.h[i]),yl=yOf(S.l[i]);
      cx.strokeStyle="#fff";cx.lineWidth=1;cx.setLineDash([]);cx.strokeRect(x-w/2-2,yh-2,w+4,yl-yh+4);
      pill("고가 "+fmtP(S.h[i]),x,yh-26,up);pill("저가 "+fmtP(S.l[i]),x,yl+8,dn)}
  }
  const d=S.d[hi_i];cx.textAlign="left";cx.textBaseline="middle";let xx=G.l+2;let yy=16;
  const put=(t,col)=>{const w=cx.measureText(t).width;if(xx+w>G.W-4&&xx>G.l+4){xx=G.l+2;yy+=17}cx.fillStyle=col;cx.fillText(t,xx,yy);xx+=w+12};
  if(S.c[hi_i]!=null){const pc=hi_i>0&&S.c[hi_i-1]?(S.c[hi_i]/S.c[hi_i-1]-1)*100:null;
    put(`${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6)}${M.tf==="W"?" (주)":M.tf==="M"?" (월)":""}`,tx);put(`시 ${fmtP(S.o[hi_i])}`,sub);put(`고 ${fmtP(S.h[hi_i])}`,sub);put(`저 ${fmtP(S.l[hi_i])}`,sub);
    put(`종 ${fmtP(S.c[hi_i])}${pc==null?"":` (${pc>0?"+":""}${pc.toFixed(1)}%)`}`,pc>=0?up:dn);
    DM.forEach(k=>{const v=S.m[k][hi_i];if(v!=null)put(`${k} ${fmtP(v)}`,MC[k])})}
}
function snap(x,y){
  const S=M.S;let i=Math.min(S.n-1,Math.max(0,iOf(x)));let p=pOf(y);
  if($("#pmag").checked&&S.c[i]!=null){let best=null,bd=14;[S.o[i],S.h[i],S.l[i],S.c[i]].forEach(v=>{const dd=Math.abs(yOf(v)-y);if(dd<bd){bd=dd;best=v}});if(best!=null)p=best}
  return{i,p};
}
function segDist(px,py,s){const dx=s.x2-s.x1,dy=s.y2-s.y1,L=dx*dx+dy*dy;let t=L?((px-s.x1)*dx+(py-s.y1)*dy)/L:0;t=Math.max(0,Math.min(1,t));return Math.hypot(px-(s.x1+t*dx),py-(s.y1+t*dy))}
function hit(x,y){let best=-1,bd=7;M.lines.forEach((L,i)=>{const s=lineXY(L);if(!s)return;const d=segDist(x,y,s);if(d<bd){bd=d;best=i}});return best}
const save=()=>{try{localStorage.setItem("maLines:"+M.code,JSON.stringify(M.lines))}catch(e){}};
const load=code=>{try{return JSON.parse(localStorage.getItem("maLines:"+code)||"[]")}catch(e){return[]}};
function setTool(t){M.tool=t;M.tmp=null;document.querySelectorAll("[data-tool]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.tool===t));cv.style.cursor=t==="cursor"?"default":"crosshair";draw()}
function setTF(tf){
  M.tf=tf;M.S=tf==="D"?daily(M.code):agg(M.code,tf);M.N=tf==="D"?120:tf==="W"?104:60;M.start=M.S.n-M.N+8;M.sel=-1;M.tmp=null;
  document.querySelectorAll("[data-tf]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.tf===tf));draw();
}
function openModal(code,tf){
  const s=D.stocks[code],i=(window.CUR||[]).indexOf(code);
  M={code,pos:i,tool:"cursor",lines:load(code),sel:-1,tmp:null,mouse:null,drag:null};
  $("#pt").textContent=s.n;$("#pm").textContent=code+" · "+(s.m==="KOSPI"?"코스피":"코스닥")+" · 시총 "+Math.round(s.cap).toLocaleString()+"억";
  $("#pb").innerHTML=[...D.mas].sort((a,b)=>b-a).map(k=>`<span class="b ${s.g[k]>0?"up":"dn"}">${k}일 ${fmt(s.g[k])}</span>`).join("");
  $("#ptoss").href="https://www.tossinvest.com/stocks/A"+code;$("#ppg").textContent=i>=0?`${i+1} / ${window.CUR.length}`:"";
  $("#ov").classList.add("on");document.body.style.overflow="hidden";
  document.querySelectorAll("[data-tool]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.tool==="cursor"));cv.style.cursor="default";
  setTF(tf||"D");
}
function closeModal(){$("#ov").classList.remove("on");document.body.style.overflow="";M=null}
function step(d){if(!M||M.pos<0)return;const n=M.pos+d;if(n<0||n>=window.CUR.length)return;openModal(window.CUR[n],M.tf)}
$("#grid").addEventListener("click",e=>{if(e.target.closest("a.tl"))return;const c=e.target.closest(".card");if(c)openModal(c.dataset.c,"D")});
$("#pclose").onclick=closeModal;$("#pprev").onclick=()=>step(-1);$("#pnext").onclick=()=>step(1);
$("#ov").addEventListener("pointerdown",e=>{if(e.target.id==="ov")closeModal()});
document.querySelectorAll("[data-tool]").forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
document.querySelectorAll("[data-tf]").forEach(b=>b.onclick=()=>{if(M)setTF(b.dataset.tf)});
$("#pundo").onclick=()=>{if(M){M.lines.pop();M.sel=-1;save();draw()}};
$("#pdel").onclick=()=>{if(M&&M.sel>=0){M.lines.splice(M.sel,1);M.sel=-1;save();draw()}};
$("#pclr").onclick=()=>{if(M&&M.lines.length&&confirm("이 종목에 그린 선을 모두 지울까?")){M.lines=[];M.sel=-1;save();draw()}};
$("#pext").onchange=()=>{if(M&&M.sel>=0&&M.lines[M.sel].t==="trend"){M.lines[M.sel].ext=$("#pext").checked;save();draw()}};
const pos=e=>{const r=cv.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top}};
function act(p){ // 클릭/탭으로 하는 동작. 선을 골랐거나 그렸으면 true
  if(M.tool==="cursor"){const h=hit(p.x,p.y);if(h>=0){M.sel=h;draw();return true}M.sel=-1;return false}
  const s=snap(p.x,p.y);
  if(M.tool==="h"){M.lines.push({t:"h",p:s.p,col:swColor});M.sel=-1;save();draw();return true}
  if(M.tool==="trend"){if(!M.tmp){M.tmp={i:s.i,p:s.p};draw()}else{if(s.i!==M.tmp.i){M.lines.push({t:"trend",d1:M.S.d[M.tmp.i],p1:M.tmp.p,d2:M.S.d[s.i],p2:s.p,ext:$("#pext").checked,col:swColor});save()}M.tmp=null;draw()}}
  return true;
}
const PT=new Map();let TP=null; // 터치: 손가락 위치들, 탭/드래그/핀치 판정
cv.addEventListener("pointerdown",e=>{if(!M)return;
  if(e.pointerType==="mouse"){if(e.button!==0)return;const p=pos(e);if(!act(p)&&M.tool==="cursor"){M.drag={x:p.x,s:M.start};draw()}return}
  try{cv.setPointerCapture(e.pointerId)}catch(_){}
  const p=pos(e);PT.set(e.pointerId,p);
  if(PT.size===1)TP={x:p.x,y:p.y,s:M.start,moved:false,pinch:false};
  else if(PT.size===2){const[a,b]=[...PT.values()];TP=TP||{};TP.pinch=true;TP.moved=true;TP.d0=Math.hypot(a.x-b.x,a.y-b.y)||1;TP.N0=M.N;
    TP.iAt=((a.x+b.x)/2-G.l)/(G.pw/M.N)+M.start-.5}
});
cv.addEventListener("pointermove",e=>{if(!M)return;
  if(e.pointerType==="mouse"){M.mouse=pos(e);if(M.drag){const dx=M.mouse.x-M.drag.x;M.start=M.drag.s-dx/(G.pw/M.N)}draw();return}
  if(!PT.has(e.pointerId))return;PT.set(e.pointerId,pos(e));
  if(PT.size>=2&&TP&&TP.pinch){const[a,b]=[...PT.values()],d=Math.hypot(a.x-b.x,a.y-b.y)||1,cxm=(a.x+b.x)/2;
    const nn=Math.max(20,Math.min(M.S.n+20,TP.N0*TP.d0/d));M.start=TP.iAt+.5-(cxm-G.l)/(G.pw/nn);M.N=nn;M.mouse=null;draw();return}
  if(TP&&PT.size===1){const p=PT.get(e.pointerId),dx=p.x-TP.x,dy=p.y-TP.y;
    if(!TP.moved&&Math.hypot(dx,dy)<8)return;TP.moved=true;M.mouse=null;M.start=TP.s-dx/(G.pw/M.N);draw()}
});
function endT(e){if(e.pointerType==="mouse")return;
  if(!PT.has(e.pointerId))return;const p=PT.get(e.pointerId);PT.delete(e.pointerId);
  if(!M||!TP)return;
  if(PT.size===0){
    if(!TP.moved&&e.type==="pointerup"){const r=act(p);M.mouse=(M.tool==="cursor"&&r)?null:p;draw()}
    TP=null}
  else if(PT.size===1){const q=[...PT.values()][0];TP={x:q.x,y:q.y,s:M.start,moved:true,pinch:false}}
}
cv.addEventListener("pointerup",endT);cv.addEventListener("pointercancel",endT);
cv.addEventListener("pointerleave",e=>{if(e.pointerType==="mouse"&&M){M.mouse=null;draw()}});
addEventListener("pointerup",e=>{if(e.pointerType==="mouse"&&M)M.drag=null});
cv.addEventListener("dblclick",()=>{if(M&&M.tool==="cursor")setTF(M.tf)});
cv.addEventListener("wheel",e=>{if(!M)return;e.preventDefault();const p=pos(e),f=e.deltaY>0?1.15:1/1.15;
  const iAt=(p.x-G.l)/(G.pw/M.N)+M.start-.5,nn=Math.max(20,Math.min(M.S.n+20,M.N*f));
  M.start=iAt+.5-(p.x-G.l)/(G.pw/nn);M.N=nn;draw()},{passive:false});
addEventListener("keydown",e=>{if(!M)return;
  if(e.key==="Escape"){if(M.tmp){M.tmp=null;draw()}else if(M.tool!=="cursor")setTool("cursor");else closeModal()}
  else if(e.key==="ArrowLeft")step(-1);else if(e.key==="ArrowRight")step(1);
  else if((e.key==="Delete"||e.key==="Backspace")&&M.sel>=0){M.lines.splice(M.sel,1);M.sel=-1;save();draw()}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){M.lines.pop();M.sel=-1;save();draw()}});
addEventListener("resize",()=>{if(M)draw()});
tabs();layout();render();layout();
</script></body></html>
"""
