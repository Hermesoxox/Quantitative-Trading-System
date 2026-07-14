"""
生成 iPhone 可用的移动端策略看板(自包含 HTML)。

产物
----
* webapp/index.html        : 完整独立页面(含 PWA 头/主题/图标)，用于 GitHub Pages，
                             iPhone Safari 打开后可"添加到主屏幕"当 App 用。
* webapp/dashboard_body.html: 仅正文(含内联样式/脚本)，用于发布为 Artifact 在手机查看。
* webapp/data.json         : 当日数据快照。

看板内容(一屏可扫)：市场状态总仓位、策略当前建议持仓(≤5只)、真实回测净值曲线、
关键绩效、免责声明。数据由一次真实回测(默认 Yahoo/东财)得到，无外部依赖、可离线看。
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import OVERLAY, PORTFOLIO
from src.factors import compute_all_factors, neutralize_factor
from src.signals import trend_filter, smooth_score
from src.portfolio import build_target_weights
from src.ml import rolling_ml_scores
from src.regime import combined_exposure
from src.backtest import Backtester, performance_summary, annual_breakdown
from examples.run_advanced import load_real, long_to_wide

# 常见蓝筹代码->简称，让看板更友好(缺失则显示代码)
NAME = {
    "600519": "贵州茅台", "601318": "中国平安", "600036": "招商银行",
    "000858": "五粮液", "600900": "长江电力", "000333": "美的集团",
    "600276": "恒瑞医药", "601166": "兴业银行", "002594": "比亚迪",
    "600030": "中信证券", "000001": "平安银行", "601888": "中国中免",
    "600887": "伊利股份", "000651": "格力电器", "600309": "万华化学",
    "601012": "隆基绿能", "600028": "中国石化", "601398": "工商银行",
    "601628": "中国人寿", "600585": "海螺水泥", "000002": "万科A",
    "600031": "三一重工", "603259": "药明康德", "601668": "中国建筑",
    "600048": "保利发展", "000725": "京东方A", "002415": "海康威视",
    "300750": "宁德时代", "601288": "农业银行", "601988": "中国银行",
    "600000": "浦发银行", "601857": "中国石油", "600104": "上汽集团",
    "601601": "中国太保", "600690": "海尔智家", "000568": "泸州老窖",
    "002475": "立讯精密", "600438": "通威股份", "601899": "紫金矿业",
    "600009": "上海机场", "000538": "云南白药", "600436": "片仔癀",
    "000063": "中兴通讯", "600745": "闻泰科技", "002241": "歌尔股份",
    "601088": "中国神华", "300059": "东方财富", "600015": "华夏银行",
    "002142": "宁波银行", "000776": "广发证券", "600745": "闻泰科技",
    "601066": "中信建投", "603501": "韦尔股份", "600406": "国电南瑞",
    "603288": "海天味业", "601336": "新华保险", "600196": "复星医药",
    "002714": "牧原股份", "601225": "陕西煤业", "600547": "山东黄金",
    "002304": "洋河股份", "603986": "兆易创新", "002241": "歌尔股份",
}


def build_data(n=30, start="2016-01-01", end=None):
    end = end or str(pd.Timestamp.today().normalize().date())
    wide, industry, log_cap, benchmark = load_real(n, start, end, "")
    close = wide["close"]
    raw = compute_all_factors(wide)
    neut = {k: neutralize_factor(v, industry, log_cap) for k, v in raw.items()}
    score, _ = rolling_ml_scores(neut, close, horizon=OVERLAY.ml_horizon,
                                 retrain_freq=OVERLAY.ml_retrain_freq,
                                 embargo_days=OVERLAY.ml_embargo_days)
    score = smooth_score(score)
    trend = trend_filter(close, 200)
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(252)
    ma60 = close.rolling(60).mean()
    expo = combined_exposure(close, benchmark=benchmark, ma_window=OVERLAY.regime_ma,
                             target_vol=OVERLAY.target_vol)
    rebal = list(close.index[::10])
    bt = Backtester(prices={k: wide[k] for k in
                    ["open", "high", "low", "close", "volume", "pre_close"]},
                    score=score, trend_pass=trend, vol=vol20, ma_exit=ma60,
                    industry=industry, rebalance_dates=rebal, init_capital=1e7,
                    exposure_schedule=expo)
    res = bt.run()
    equity = res["equity"]
    last_close = close.iloc[-1]

    # 当前建议持仓 = 最新截面的目标前5(不受当日仓位影响)。
    # 仓位由 regime 单独决定：即便建议空仓，这几只也是"转多后的首选清单"。
    target_w = build_target_weights(score.iloc[-1], trend.iloc[-1],
                                    vol20.iloc[-1], industry)
    holdings = []
    for code, w in target_w.sort_values(ascending=False).items():
        p = float(last_close.get(code, np.nan))
        if np.isfinite(p):
            holdings.append({"code": code, "name": NAME.get(code, code),
                             "weight": float(w), "price": round(p, 2)})

    exposure = float(expo.iloc[-1])
    if exposure < 0.10:
        regime = {"label": "空仓避险", "icon": "❄️", "cls": "off"}
    elif exposure < 0.6:
        regime = {"label": "减仓谨慎", "icon": "⛅", "cls": "half"}
    else:
        regime = {"label": "可持仓", "icon": "☀️", "cls": "on"}
    regime["exposure"] = round(exposure * 100)

    perf = performance_summary(equity, res["trades"], res["turnover"])
    ann = annual_breakdown(equity)
    years = [{"year": int(y), "ret": float(r)}
             for y, r in ann["return"].items()]

    # 净值曲线降采样到 ~120 点
    eqn = equity / equity.iloc[0]
    step = max(1, len(eqn) // 120)
    pts = [{"t": d.strftime("%Y-%m-%d"), "v": round(float(v), 4)}
           for d, v in eqn.iloc[::step].items()]

    return {
        "as_of": str(equity.index.max().date()),
        "universe": n,
        "regime": regime,
        "holdings": holdings,
        "metrics": {k: perf.get(k) for k in
                    ["annual_return", "max_drawdown", "sharpe", "calmar",
                     "profit_loss_ratio", "annual_turnover"]},
        "years": years,
        "equity": pts,
    }


# ---------------------------------------------------------------- HTML 渲染
def svg_equity(pts, w=440, h=140, pad=6):
    if len(pts) < 2:
        return ""
    vs = [p["v"] for p in pts]
    lo, hi = min(vs), max(vs)
    rng = (hi - lo) or 1.0
    n = len(pts)
    def X(i): return pad + i * (w - 2 * pad) / (n - 1)
    def Y(v): return pad + (h - 2 * pad) * (1 - (v - lo) / rng)
    line = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                    for i, v in enumerate(vs))
    area = (f"M{X(0):.1f},{h - pad:.1f} "
            + " ".join(f"L{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vs))
            + f" L{X(n - 1):.1f},{h - pad:.1f} Z")
    ex, ey = X(n - 1), Y(vs[-1])
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" preserveAspectRatio="none" '
            f'role="img" aria-label="策略净值曲线">'
            f'<path d="{area}" class="sp-area"/>'
            f'<path d="{line}" class="sp-line"/>'
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.4" class="sp-dot"/></svg>')


def pct(x, digits=1):
    return "—" if x is None else f"{x*100:.{digits}f}%"


def render_body(d):
    r = d["regime"]
    m = d["metrics"]
    hold_rows = "".join(
        f'<li><span class="h-name">{h["name"]}<em>{h["code"]}</em></span>'
        f'<span class="h-bar"><i style="width:{max(6,h["weight"]*100):.0f}%"></i></span>'
        f'<span class="h-w num">{h["weight"]*100:.0f}%</span>'
        f'<span class="h-p num">{h["price"]:.2f}</span></li>'
        for h in d["holdings"]) or '<li class="empty">当前建议空仓（市场状态转弱）</li>'
    year_rows = "".join(
        f'<span class="yr"><b>{y["year"]}</b>'
        f'<i class="{"up" if y["ret"]>=0 else "down"}">{y["ret"]*100:+.0f}%</i></span>'
        for y in d["years"] if y["year"] >= 2017)

    ar = m["annual_return"]
    dd = m["max_drawdown"]
    return f"""
<style>
:root {{
  --bg:#f4f4f2; --card:#ffffff; --ink:#14171c; --muted:#6a7078; --line:#e7e7e3;
  --accent:#2f5d7c; --accent-soft:#2f5d7c22; --up:#d0342c; --down:#159a52;
  --on:#159a52; --half:#c9871f; --off:#8a9099; --shadow:0 1px 3px #0000000f;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0e1013; --card:#171a20; --ink:#e9ebef; --muted:#949aa3;
    --line:#242830; --accent:#79b4d8; --accent-soft:#79b4d81f; --up:#f2564a;
    --down:#3cc07d; --on:#3cc07d; --half:#e0a63e; --off:#7d848d; --shadow:none; }}
}}
:root[data-theme="dark"] {{ --bg:#0e1013; --card:#171a20; --ink:#e9ebef; --muted:#949aa3;
  --line:#242830; --accent:#79b4d8; --accent-soft:#79b4d81f; --up:#f2564a;
  --down:#3cc07d; --on:#3cc07d; --half:#e0a63e; --off:#7d848d; --shadow:none; }}
:root[data-theme="light"] {{ --bg:#f4f4f2; --card:#ffffff; --ink:#14171c; --muted:#6a7078;
  --line:#e7e7e3; --accent:#2f5d7c; --accent-soft:#2f5d7c22; --up:#d0342c;
  --down:#159a52; --on:#159a52; --half:#c9871f; --off:#8a9099; --shadow:0 1px 3px #0000000f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.45; }}
.num {{ font-variant-numeric:tabular-nums; font-feature-settings:"tnum"; }}
.wrap {{ max-width:480px; margin:0 auto; padding:0 16px 40px; }}
header {{ position:sticky; top:0; z-index:5; background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(10px); border-bottom:1px solid var(--line);
  padding:14px 16px calc(12px + env(safe-area-inset-top)); margin-bottom:14px; }}
header .wrap {{ padding:0; display:flex; align-items:baseline; justify-content:space-between; }}
.brand {{ font-weight:700; font-size:17px; letter-spacing:.2px; }}
.brand span {{ color:var(--accent); }}
.date {{ color:var(--muted); font-size:12.5px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:16px; margin-bottom:14px; box-shadow:var(--shadow); }}
.eyebrow {{ font-size:11px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); margin:0 0 10px; font-weight:600; }}
/* regime hero */
.hero {{ display:flex; align-items:center; gap:16px; }}
.hero .dot {{ width:14px; height:14px; border-radius:50%; flex:none; }}
.hero.on .dot {{ background:var(--on); box-shadow:0 0 0 5px var(--on)22; }}
.hero.half .dot {{ background:var(--half); box-shadow:0 0 0 5px var(--half)22; }}
.hero.off .dot {{ background:var(--off); box-shadow:0 0 0 5px var(--off)22; }}
.hero .lab {{ font-size:20px; font-weight:700; }}
.hero .sub {{ color:var(--muted); font-size:12.5px; margin-top:2px; }}
.hero .expo {{ margin-left:auto; text-align:right; }}
.hero .expo b {{ font-size:30px; font-weight:750; letter-spacing:-.5px; }}
.hero .expo em {{ display:block; color:var(--muted); font-size:11px; font-style:normal; }}
/* holdings */
ul.holds {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:2px; }}
ul.holds li {{ display:grid; grid-template-columns:1fr 68px 40px 52px; align-items:center;
  gap:8px; padding:9px 0; border-bottom:1px solid var(--line); }}
ul.holds li:last-child {{ border-bottom:0; }}
li.empty {{ display:block!important; color:var(--muted); text-align:center; padding:18px 0; }}
.h-name {{ font-weight:600; font-size:15px; }}
.h-name em {{ display:block; font-style:normal; color:var(--muted); font-size:11px;
  font-variant-numeric:tabular-nums; letter-spacing:.02em; }}
.h-bar {{ height:6px; background:var(--accent-soft); border-radius:99px; overflow:hidden; }}
.h-bar i {{ display:block; height:100%; background:var(--accent); border-radius:99px; }}
.h-w {{ text-align:right; font-weight:650; font-size:14px; }}
.h-p {{ text-align:right; color:var(--muted); font-size:13px; }}
/* chart */
.spark {{ width:100%; height:140px; display:block; }}
.sp-area {{ fill:var(--accent-soft); stroke:none; }}
.sp-line {{ fill:none; stroke:var(--accent); stroke-width:2; stroke-linejoin:round; stroke-linecap:round; vector-effect:non-scaling-stroke; }}
.sp-dot {{ fill:var(--accent); stroke:var(--card); stroke-width:2; }}
.chart-cap {{ display:flex; justify-content:space-between; color:var(--muted); font-size:11.5px; margin-top:6px; }}
/* metrics */
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line);
  border-radius:12px; overflow:hidden; }}
.grid .m {{ background:var(--card); padding:13px 14px; }}
.grid .m .v {{ font-size:21px; font-weight:730; letter-spacing:-.3px; }}
.grid .m .k {{ color:var(--muted); font-size:11.5px; margin-top:2px; }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }}
/* year chips */
.years {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }}
.yr {{ display:flex; flex-direction:column; align-items:center; gap:1px; padding:6px 9px;
  background:var(--bg); border:1px solid var(--line); border-radius:9px; min-width:52px; }}
.yr b {{ font-size:11px; color:var(--muted); font-weight:600; }}
.yr i {{ font-style:normal; font-weight:700; font-size:13px; font-variant-numeric:tabular-nums; }}
.foot {{ color:var(--muted); font-size:11.5px; line-height:1.6; margin-top:6px; }}
.foot b {{ color:var(--ink); }}
</style>
<header><div class="wrap"><div class="brand">量化<span>罗盘</span></div>
<div class="date num">{d['as_of']} · {d['universe']}只蓝筹池</div></div></header>
<main class="wrap">
  <section class="card">
    <p class="eyebrow">市场状态 · 建议总仓位</p>
    <div class="hero {r['cls']}">
      <span class="dot"></span>
      <div><div class="lab">{r['icon']} {r['label']}</div>
        <div class="sub">依大盘趋势与波动率动态调节仓位</div></div>
      <div class="expo"><b class="num">{r['exposure']}%</b><em>目标仓位</em></div>
    </div>
  </section>

  <section class="card">
    <p class="eyebrow">{'候选清单 · 转多后首选（当前建议空仓）' if r['cls']=='off' else '策略当前建议持仓'} · 最多{PORTFOLIO.n_holdings_max}只</p>
    <ul class="holds">{hold_rows}</ul>
  </section>

  <section class="card">
    <p class="eyebrow">策略净值（真实回测，起点=1.0）</p>
    {svg_equity(d['equity'])}
    <div class="chart-cap"><span>{d['equity'][0]['t'] if d['equity'] else ''}</span>
      <span>净值 {d['equity'][-1]['v'] if d['equity'] else ''}×</span>
      <span>{d['equity'][-1]['t'] if d['equity'] else ''}</span></div>
    <div class="years">{year_rows}</div>
  </section>

  <section class="card">
    <p class="eyebrow">关键绩效（全区间真实回测）</p>
    <div class="grid">
      <div class="m"><div class="v {'up' if (ar or 0)>=0 else 'down'} num">{pct(ar)}</div><div class="k">年化收益</div></div>
      <div class="m"><div class="v down num">{pct(dd)}</div><div class="k">最大回撤</div></div>
      <div class="m"><div class="v num">{m['sharpe']}</div><div class="k">夏普比率</div></div>
      <div class="m"><div class="v num">{m['profit_loss_ratio']}</div><div class="k">盈亏比</div></div>
    </div>
  </section>

  <section class="card">
    <p class="foot"><b>免责声明</b> 本页为量化研究/辅助决策工具，<b>非投资建议</b>，
    不自动下单。数据经真实回测生成；实盘表现通常显著低于回测(约打6–7折)，
    且纯样本外年化仅约个位数。请务必先用模拟盘/小资金验证，风险自负。</p>
    <p class="foot">数据源：Yahoo/东财公开行情 · 策略：多因子+市场状态+回撤守卫 · ≤5只集中持仓</p>
  </section>
</main>"""


PWA_HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="量化罗盘">
<meta name="theme-color" content="#0e1013" media="(prefers-color-scheme:dark)">
<meta name="theme-color" content="#f4f4f2" media="(prefers-color-scheme:light)">
<title>量化罗盘 · A股策略看板</title>
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Crect width='180' height='180' rx='40' fill='%230e1013'/%3E%3Ctext x='90' y='120' font-size='96' text-anchor='middle'%3E%F0%9F%A7%AD%3C/text%3E%3C/svg%3E">"""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"生成移动端看板… 股票池={n}")
    d = build_data(n)
    body = render_body(d)

    out = os.path.join(os.path.dirname(__file__), "..", "webapp")
    os.makedirs(out, exist_ok=True)
    # 完整独立页(GitHub Pages / 添加到主屏幕)
    full = (f"<!doctype html><html lang=\"zh-CN\"><head>{PWA_HEAD}</head>"
            f"<body>{body}</body></html>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(full)
    # 仅正文(用于发布 Artifact)
    with open(os.path.join(out, "dashboard_body.html"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print("已生成 webapp/index.html, dashboard_body.html, data.json")
    print(f"  日期={d['as_of']}  仓位={d['regime']['exposure']}%  "
          f"持仓{len(d['holdings'])}只  年化={pct(d['metrics']['annual_return'])}")


if __name__ == "__main__":
    main()
