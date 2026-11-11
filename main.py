"""
AI株価シグナル Ver.1（完全無料版）

日本株・米国株の日足を取得し、MACD・RSI・移動平均線・出来高からスコアを計算して、
買い候補・売り/警戒候補を Telegram に通知する。注文は一切しない。

使い方:
    python main.py                  # 全銘柄を分析して通知
    python main.py --market jp      # 日本株だけ
    python main.py --market us      # 米国株だけ
    python main.py --dry-run        # 通知せず画面に表示だけ
    python main.py --telegram-test  # Telegram の接続テスト
    python main.py --report         # 仮想売買の成績を表示

環境変数:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID（未設定なら画面表示のみ）
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

import ai_analysis

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
WATCHLIST_PATH = ROOT / "watchlist.csv"
DB_PATH = ROOT / "data" / "signals.db"

DEFAULT_CONFIG = {
    "buy_score": 75,
    "stop_loss_pct": -5.0,
    "take_profit_pct": 15.0,
    "volume_surge": 1.5,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "cross_lookback_days": 3,
    "earnings_warning_days": 7,
    "notify_when_no_signal": True,
    "show_news": True,
    "paper_trading": True,
    "ai_enabled": True,
    "ai_models": ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash"],
    "ai_web_search": True,
    "ai_max_calls": 8,
    "ai_interval_sec": 8,
    "ai_min_score": 25,
}

DISCLAIMER = "※機械的な条件判定の結果です。売買の推奨ではありません。最終判断はご自身で。"


# ---------------------------------------------------------------- 設定・銘柄
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return cfg


@dataclass
class Stock:
    ticker: str
    name: str
    market: str  # JP / US
    sector: str = ""
    entry_price: float | None = None  # 実際に保有している場合の取得単価
    memo: str = ""


def load_watchlist() -> list[Stock]:
    # Excel で保存すると Shift_JIS になることがあるので両方試す
    for enc in ("utf-8-sig", "cp932"):
        try:
            with open(WATCHLIST_PATH, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError("watchlist.csv の文字コードを読めません（UTF-8で保存してください）")

    stocks = []
    for r in rows:
        ticker = (r.get("ticker") or "").strip()
        if not ticker or ticker.startswith("#"):
            continue
        entry = (r.get("entry_price") or "").strip()
        stocks.append(
            Stock(
                ticker=ticker,
                name=(r.get("name") or ticker).strip(),
                market=(r.get("market") or ("JP" if ticker.endswith(".T") else "US")).strip().upper(),
                sector=(r.get("sector") or "").strip().lower(),
                entry_price=float(entry) if entry else None,
                memo=(r.get("memo") or "").strip(),
            )
        )
    return stocks


# ---------------------------------------------------------------- データ取得
MARKET_CLOSE = {"JP": ("Asia/Tokyo", dt.time(15, 45)), "US": ("America/New_York", dt.time(16, 15))}


def fetch_prices(ticker: str, market: str = "US") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError("株価データを取得できませんでした")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    # 取引時間中に実行した場合、今日の足は未確定（出来高が少ない等）なので除く
    tz, close_t = MARKET_CLOSE.get(market, MARKET_CLOSE["US"])
    now = dt.datetime.now(ZoneInfo(tz))
    if len(df) and df.index[-1].date() == now.date() and now.time() < close_t:
        df = df.iloc[:-1]
    if len(df) < 80:
        raise RuntimeError(f"データが足りません（{len(df)}日分）")
    return df


def fetch_earnings_date(ticker: str) -> dt.date | None:
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None
        d = dates[0]
        return d if isinstance(d, dt.date) else pd.Timestamp(d).date()
    except Exception:
        return None


def fetch_headlines(ticker: str, limit: int = 2) -> list[str]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    titles = []
    for it in items:
        title = it.get("title") or (it.get("content") or {}).get("title")
        if title:
            titles.append(title.strip())
        if len(titles) >= limit:
            break
    return titles


# ---------------------------------------------------------------- 指標計算
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["sma25"] = close.rolling(25).mean()
    df["sma75"] = close.rolling(75).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - 100 / (1 + rs)

    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean().shift(1)
    return df


def days_since_cross(a: pd.Series, b: pd.Series, lookback: int, up: bool) -> int | None:
    """直近 lookback 本以内に a が b を上抜け(up=True)/下抜けした日が何本前か。無ければ None。"""
    diff = (a - b).dropna()
    for i in range(1, min(lookback, len(diff) - 1) + 1):
        now, prev = diff.iloc[-i], diff.iloc[-i - 1]
        if up and now > 0 >= prev:
            return i - 1
        if not up and now < 0 <= prev:
            return i - 1
    return None


# ---------------------------------------------------------------- 判定
@dataclass
class Result:
    stock: Stock
    date: dt.date
    close: float
    rsi: float
    macd: float
    macd_signal: float
    vol_ratio: float
    score: int
    breakdown: dict
    signal: str  # BUY / SELL / WATCH / NONE
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    ai: "ai_analysis.AIResult | None" = None
    ai_error: str = ""

    @property
    def total(self) -> int:
        """総合スコア（100点）：テクニカル50% ＋ AI(業績25・ニュース15・リスク10)。AI無しならテクニカルのみ。"""
        return round(self.score * 0.5 + self.ai.total) if self.ai else self.score

    def technical_text(self) -> str:
        b = self.breakdown
        return (
            f"日付 {self.date} 終値 {self.close:.2f} / テクニカルスコア {self.score}/100"
            f"（トレンド{b['トレンド']}/30, MACD{b['MACD']}/30, RSI{b['RSI']}/20, 出来高{b['出来高']}/20）\n"
            f"RSI {self.rsi:.1f} / MACD {self.macd:.3f} シグナル {self.macd_signal:.3f} / 出来高 平均比{self.vol_ratio:.2f}倍\n"
            f"判定根拠: {' / '.join(self.reasons) or 'なし'}\n注意: {' / '.join(self.warnings) or 'なし'}"
        )


def score_text(r: Result) -> str:
    if r.ai:
        return f"総合 {r.total}/100（テクニカル{r.score}・AI{r.ai.total}/50）"
    return f"テクニカル {r.score}/100"


def ai_lines(r: Result) -> list[str]:
    if r.ai:
        a = r.ai
        out = [
            f"🤖 AI評価：{a.stance}（業績{a.fundamental}/25・材料{a.news}/15・リスク{a.risk}/10）",
            f"   {a.summary}",
        ]
        out += [f"   ＋ {c}" for c in a.catalysts]
        out += [f"   － {x}" for x in a.risks]
        return out
    if r.ai_error:
        return [f"🤖 AI評価：失敗（{r.ai_error[:80]}）"]
    return []


def apply_ai(results: list[Result], cfg: dict) -> None:
    """買い・売り・監視・保有中の銘柄だけAIで評価（無料枠の回数制限のため）。"""
    if not cfg.get("ai_enabled") or not ai_analysis.api_key():
        print("（GEMINI_API_KEY 未設定のためAI分析はスキップ）")
        return
    prio = {"BUY": 0, "SELL": 1, "WATCH": 2}
    targets = sorted(
        [r for r in results if r.signal in prio or r.stock.entry_price],
        key=lambda r: (prio.get(r.signal, 3), -r.score),
    )[: int(cfg["ai_max_calls"])]
    for i, r in enumerate(targets):
        if i:
            time.sleep(float(cfg["ai_interval_sec"]))
        try:
            r.ai = ai_analysis.analyze(r.stock, r.technical_text(), cfg)
            print(f"  AI {r.stock.ticker}: {r.ai.stance} {r.ai.total}/50 ({r.ai.model})")
        except Exception as e:
            r.ai_error = str(e)
            print(f"  AI {r.stock.ticker}: {e}")
            continue
        # AIだけで買いにはしない。テクニカル買い候補をAIが否定したら「監視」に下げる。
        if r.signal == "BUY" and (r.ai.total < cfg["ai_min_score"] or r.ai.stance == "弱気"):
            r.signal = "WATCH"
            r.reasons.append(f"AI評価が低いため監視に変更（{r.ai.total}/50・{r.ai.stance}）")


def analyze(stock: Stock, df: pd.DataFrame, cfg: dict) -> Result:
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    lookback = int(cfg["cross_lookback_days"])
    reasons: list[str] = []
    sell: list[str] = []
    warnings: list[str] = []

    # --- トレンド 30点
    trend = 0
    if last["Close"] > last["sma25"]:
        trend += 10
        reasons.append("株価が25日線の上")
    if last["sma25"] > last["sma75"]:
        trend += 10
        reasons.append("25日線 > 75日線（上昇トレンド）")
    if last["sma25"] > df["sma25"].iloc[-6]:
        trend += 10

    # --- MACD 30点
    macd_pts = 0
    gc = days_since_cross(df["macd"], df["macd_signal"], lookback, up=True)
    dc = days_since_cross(df["macd"], df["macd_signal"], lookback, up=False)
    if last["macd"] > last["macd_signal"]:
        macd_pts += 15
    if gc is not None:
        macd_pts += 15
        reasons.append("MACDゴールデンクロス" + ("（本日）" if gc == 0 else f"（{gc}日前）"))
    if dc is not None:
        sell.append("MACDデッドクロス" + ("（本日）" if dc == 0 else f"（{dc}日前）"))

    # --- RSI 20点
    rsi = float(last["rsi"])
    if 40 <= rsi < 65:
        rsi_pts = 20
    elif cfg["rsi_oversold"] <= rsi < 40:
        rsi_pts = 15
        reasons.append(f"RSI {rsi:.0f}（売られすぎからの戻り圏）")
    elif 65 <= rsi < cfg["rsi_overbought"]:
        rsi_pts = 10
    elif rsi < cfg["rsi_oversold"]:
        rsi_pts = 10
        warnings.append(f"RSI {rsi:.0f}：売られすぎ（下落継続に注意）")
    else:
        rsi_pts = 0
        sell.append(f"RSI {rsi:.0f}：買われすぎ")

    # --- 出来高 20点
    vr = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 1.0
    up_day = last["Close"] >= prev["Close"]
    if vr >= cfg["volume_surge"] and up_day:
        vol_pts = 20
        reasons.append(f"出来高 平均比{vr:.1f}倍で上昇（資金流入）")
    elif vr >= cfg["volume_surge"] and not up_day:
        vol_pts = 0
        sell.append(f"出来高 平均比{vr:.1f}倍で下落（売り圧力）")
    elif vr >= 1.2:
        vol_pts = 10
    else:
        vol_pts = 5

    # --- 25日線割れ
    if last["Close"] < last["sma25"] and prev["Close"] >= prev["sma25"]:
        sell.append("25日線を下抜け")

    # --- 保有中の損切り・利確（watchlist の entry_price）
    if stock.entry_price:
        pnl = (last["Close"] / stock.entry_price - 1) * 100
        if pnl <= cfg["stop_loss_pct"]:
            sell.insert(0, f"損切り目安に到達：取得単価比 {pnl:+.1f}%")
        elif pnl >= cfg["take_profit_pct"]:
            sell.insert(0, f"利益確定目安に到達：取得単価比 {pnl:+.1f}%")
        else:
            warnings.append(f"保有中：取得単価比 {pnl:+.1f}%")

    # --- イベントリスク
    if stock.sector in ("bio", "biotech", "バイオ"):
        warnings.append("バイオ株：治験結果・FDA関連ニュースで急変動の恐れ")
    ed = fetch_earnings_date(stock.ticker)
    today = df.index[-1].date()
    if ed and 0 <= (ed - today).days <= cfg["earnings_warning_days"]:
        warnings.append(f"決算発表予定 {ed:%m/%d}（発表前後は値動きが荒くなりやすい）")
    stale = (dt.date.today() - today).days
    if stale > 5:
        warnings.append(f"データが {stale} 日前のもの（休場・上場廃止等を確認）")

    score = trend + macd_pts + rsi_pts + vol_pts
    breakdown = {"トレンド": trend, "MACD": macd_pts, "RSI": rsi_pts, "出来高": vol_pts}

    # 買い候補は「スコア」と「明確な条件（MACDが上向き）」の両方を満たす場合のみ
    if sell:
        signal = "SELL"
        reasons = sell
    elif score >= cfg["buy_score"] and last["macd"] > last["macd_signal"]:
        signal = "BUY"
    elif score >= cfg["buy_score"] - 10:
        signal = "WATCH"
    else:
        signal = "NONE"

    return Result(
        stock=stock,
        date=today,
        close=float(last["Close"]),
        rsi=rsi,
        macd=float(last["macd"]),
        macd_signal=float(last["macd_signal"]),
        vol_ratio=vr,
        score=int(score),
        breakdown=breakdown,
        signal=signal,
        reasons=reasons,
        warnings=warnings,
    )


# ---------------------------------------------------------------- 保存
def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis (
            date TEXT, ticker TEXT, name TEXT, market TEXT,
            close REAL, rsi REAL, macd REAL, macd_signal REAL, vol_ratio REAL,
            score INTEGER, breakdown TEXT, signal TEXT, reasons TEXT, warnings TEXT,
            run_at TEXT,
            PRIMARY KEY (date, ticker)
        );
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, name TEXT, market TEXT,
            entry_date TEXT, entry_price REAL, entry_score INTEGER,
            exit_date TEXT, exit_price REAL, pnl_pct REAL, exit_reason TEXT
        );
        CREATE TABLE IF NOT EXISTS ai_analysis (
            date TEXT, ticker TEXT, model TEXT,
            fundamental INTEGER, news INTEGER, risk INTEGER, ai_total INTEGER, total INTEGER,
            stance TEXT, summary TEXT, catalysts TEXT, risks TEXT,
            PRIMARY KEY (date, ticker)
        );
        """
    )
    return con


def save_ai(con: sqlite3.Connection, r: Result) -> None:
    if not r.ai:
        return
    a = r.ai
    con.execute(
        "INSERT OR REPLACE INTO ai_analysis VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            r.date.isoformat(), r.stock.ticker, a.model, a.fundamental, a.news, a.risk, a.total, r.total,
            a.stance, a.summary, json.dumps(a.catalysts, ensure_ascii=False), json.dumps(a.risks, ensure_ascii=False),
        ),
    )


def save_result(con: sqlite3.Connection, r: Result) -> None:
    con.execute(
        "INSERT OR REPLACE INTO analysis VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            r.date.isoformat(), r.stock.ticker, r.stock.name, r.stock.market,
            r.close, r.rsi, r.macd, r.macd_signal, r.vol_ratio,
            r.score, json.dumps(r.breakdown, ensure_ascii=False), r.signal,
            " / ".join(r.reasons), " / ".join(r.warnings),
            dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def update_paper_trade(con: sqlite3.Connection, r: Result, cfg: dict) -> str | None:
    """仮想売買：買いシグナルの終値で仮想購入、売りシグナル・損切り・利確で仮想売却。"""
    d = r.date.isoformat()
    row = con.execute(
        "SELECT id, entry_price FROM paper_trades WHERE ticker=? AND exit_date IS NULL",
        (r.stock.ticker,),
    ).fetchone()
    if row:
        trade_id, entry = row
        pnl = (r.close / entry - 1) * 100
        reason = None
        if pnl <= cfg["stop_loss_pct"]:
            reason = "損切り"
        elif pnl >= cfg["take_profit_pct"]:
            reason = "利益確定"
        elif r.signal == "SELL":
            reason = "売りシグナル"
        if reason:
            con.execute(
                "UPDATE paper_trades SET exit_date=?, exit_price=?, pnl_pct=?, exit_reason=? WHERE id=?",
                (d, r.close, pnl, reason, trade_id),
            )
            return f"仮想売却 {r.stock.ticker}：{reason} {pnl:+.1f}%"
        return None
    if r.signal == "BUY":
        closed_today = con.execute(
            "SELECT 1 FROM paper_trades WHERE ticker=? AND exit_date=?", (r.stock.ticker, d)
        ).fetchone()
        if not closed_today:
            con.execute(
                "INSERT INTO paper_trades (ticker,name,market,entry_date,entry_price,entry_score) VALUES (?,?,?,?,?,?)",
                (r.stock.ticker, r.stock.name, r.stock.market, d, r.close, r.score),
            )
            return f"仮想購入 {r.stock.ticker} @ {r.close:,.2f}"
    return None


def paper_report(con: sqlite3.Connection) -> str:
    closed = pd.read_sql_query(
        "SELECT * FROM paper_trades WHERE exit_date IS NOT NULL ORDER BY exit_date, id", con
    )
    open_n = con.execute("SELECT COUNT(*) FROM paper_trades WHERE exit_date IS NULL").fetchone()[0]
    if closed.empty:
        return f"仮想売買：決済済み 0件（保有中 {open_n}件）"
    wins = (closed["pnl_pct"] > 0).sum()
    equity = (1 + closed["pnl_pct"] / 100).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min() * 100
    return (
        f"仮想売買：決済 {len(closed)}件 / 保有中 {open_n}件\n"
        f"  勝率 {wins / len(closed) * 100:.0f}%  平均損益 {closed['pnl_pct'].mean():+.1f}%\n"
        f"  累積損益 {(equity.iloc[-1] - 1) * 100:+.1f}%  最大ドローダウン {max_dd:.1f}%"
    )


# ---------------------------------------------------------------- 通知
def fmt_price(r: Result) -> str:
    return f"{r.close:,.0f}円" if r.stock.market == "JP" else f"${r.close:,.2f}"


def build_message(results: list[Result], errors: list[str], trade_logs: list[str], report: str, market: str, cfg: dict) -> str | None:
    buys = sorted([r for r in results if r.signal == "BUY"], key=lambda r: -r.total)
    sells = [r for r in results if r.signal == "SELL"]
    watches = sorted([r for r in results if r.signal == "WATCH"], key=lambda r: -r.total)
    if not buys and not sells and not cfg["notify_when_no_signal"] and not errors:
        return None

    label = {"jp": "日本株", "us": "米国株"}.get(market, "日本株・米国株")
    day = max((r.date for r in results), default=dt.date.today())
    lines = [f"📊 AI株価シグナル {day:%Y-%m-%d}（{label}）", f"分析 {len(results)}銘柄"]

    def block(r: Result, head: str) -> None:
        lines.append("")
        lines.append(f"{head} {r.stock.ticker} {r.stock.name}  {score_text(r)}")
        lines.append(f"  終値 {fmt_price(r)} / RSI {r.rsi:.0f} / 出来高 平均比{r.vol_ratio:.1f}倍")
        for s in r.reasons:
            lines.append(f"  ・{s}")
        for w in r.warnings:
            lines.append(f"  ⚠ {w}")
        lines.extend(f"  {x}" for x in ai_lines(r))
        for h in r.headlines:
            lines.append(f"  📰 {h}")

    if buys:
        lines.append("\n【買い候補】")
        for r in buys:
            block(r, "🟢")
            lines.append(f"  損切り目安 {r.close * (1 + cfg['stop_loss_pct'] / 100):,.2f}（{cfg['stop_loss_pct']:+.0f}%）")
    if sells:
        lines.append("\n【売り・警戒】")
        for r in sells:
            block(r, "🔴")
    if watches:
        lines.append("\n【もう少しで買い条件（監視）】")
        for r in watches:
            lines.append(f"🟡 {r.stock.ticker} {r.stock.name} {score_text(r)}")
            if r.ai:
                lines.append(f"   🤖 {r.ai.stance}：{r.ai.summary}")
    if not buys and not sells:
        lines.append("\n本日は条件に合う銘柄はありませんでした。")
    if trade_logs:
        lines.append("\n【仮想売買】")
        lines += [f"  {t}" for t in trade_logs]
    if report:
        lines.append("")
        lines.append(report)
    if errors:
        lines.append("\n【取得エラー】")
        lines += [f"  {e}" for e in errors]
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


MARKET_LABEL = {"JP": "日本株", "US": "米国株"}
SIGNAL_LABEL = {"BUY": "🟢 買い候補", "SELL": "🔴 売り・警戒", "WATCH": "🟡 監視", "NONE": "－"}


def build_markdown(results: list[Result], trade_logs: dict[str, str], report: str, errors: list[str], market: str, cfg: dict) -> str:
    """スマホの GitHub で読みやすい結果ページ（Markdown）。"""
    label = MARKET_LABEL.get(market, market)
    day = max((r.date for r in results), default=dt.date.today())
    jst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    out = [
        f"# {label} 分析結果 {day:%Y-%m-%d}",
        "",
        f"更新：{jst:%Y-%m-%d %H:%M}（日本時間）／ 分析 {len(results)}銘柄",
        "",
    ]

    def section(title: str, rows: list[Result], with_stop: bool = False) -> None:
        out.append(f"## {title}")
        out.append("")
        if not rows:
            out.append("なし")
            out.append("")
            return
        for r in rows:
            out.append(f"### {r.stock.ticker} {r.stock.name} ― {score_text(r)}")
            out.append("")
            out.append(f"- 終値 {fmt_price(r)} ／ RSI {r.rsi:.0f} ／ 出来高 平均比{r.vol_ratio:.1f}倍")
            for s in r.reasons:
                out.append(f"- {s}")
            for w in r.warnings:
                out.append(f"- ⚠ {w}")
            if r.ai:
                a = r.ai
                out.append(f"- 🤖 **AI評価：{a.stance}**（業績{a.fundamental}/25・材料{a.news}/15・リスク{a.risk}/10）{a.summary}")
                out.extend(f"  - ＋ {c}" for c in a.catalysts)
                out.extend(f"  - － {x}" for x in a.risks)
            elif r.ai_error:
                out.append(f"- 🤖 AI評価：失敗（{r.ai_error[:80]}）")
            if with_stop:
                out.append(f"- 損切り目安 {r.close * (1 + cfg['stop_loss_pct'] / 100):,.2f}（{cfg['stop_loss_pct']:+.0f}%）")
            for h in r.headlines:
                out.append(f"- 📰 {h}")
            out.append("")

    section("🟢 買い候補", sorted([r for r in results if r.signal == "BUY"], key=lambda r: -r.total), with_stop=True)
    section("🔴 売り・警戒", [r for r in results if r.signal == "SELL"])
    section("🟡 監視（もう少しで買い条件）", sorted([r for r in results if r.signal == "WATCH"], key=lambda r: -r.total))

    out += [
        "## 全銘柄一覧（総合スコア順）", "",
        "| 銘柄 | 判定 | 総合 | テクニカル | AI | 終値 | RSI | 出来高比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: -r.total):
        ai = f"{r.ai.total}/50 {r.ai.stance}" if r.ai else "－"
        out.append(
            f"| {r.stock.ticker} {r.stock.name} | {SIGNAL_LABEL[r.signal]} | {r.total} | {r.score} | {ai} "
            f"| {fmt_price(r)} | {r.rsi:.0f} | {r.vol_ratio:.1f} |"
        )
    out += ["", "総合＝テクニカル×0.5＋AI（業績25・材料15・リスク10）。AI未評価の銘柄はテクニカルのみ。"]
    out.append("")

    out += ["## 仮想売買", ""]
    logs = [t for tk, t in trade_logs.items() if any(r.stock.ticker == tk for r in results)]
    out += [f"- {t}" for t in logs] or ["- 本日の売買なし"]
    out += ["", "```", report, "```", ""]
    if errors:
        out += ["## 取得エラー", ""] + [f"- {e}" for e in errors] + [""]
    out += ["---", "", DISCLAIMER, ""]
    return "\n".join(out)


def write_result_pages(results: list[Result], trade_logs: dict[str, str], report: str, errors: list[str], cfg: dict) -> None:
    out_dir = ROOT / "results"
    archive = out_dir / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    for mk in ("JP", "US"):
        subset = [r for r in results if r.stock.market == mk]
        errs = [e for e in errors if (e.split(":")[0].endswith(".T")) == (mk == "JP")]
        if not subset and not errs:
            continue
        md = build_markdown(subset, trade_logs, report, errs, mk, cfg)
        name = "最新_日本株.md" if mk == "JP" else "最新_米国株.md"
        (out_dir / name).write_text(md, encoding="utf-8")
        day = max((r.date for r in subset), default=dt.date.today())
        (archive / f"{day:%Y-%m-%d}_{mk.lower()}.md").write_text(md, encoding="utf-8")
        print(f"結果ページを保存: results/{name}")


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("（TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未設定のため通知はスキップ）")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram は1通4096文字まで
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > 3900:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    ok = True
    for c in chunks:
        res = requests.post(url, data={"chat_id": chat_id, "text": c, "disable_web_page_preview": True}, timeout=20)
        if res.status_code != 200:
            print(f"Telegram送信エラー: {res.status_code} {res.text[:200]}")
            ok = False
    return ok


# ---------------------------------------------------------------- メイン
def main() -> int:
    ap = argparse.ArgumentParser(description="AI株価シグナル Ver.1")
    ap.add_argument("--market", choices=["jp", "us", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="通知せず画面に表示だけ")
    ap.add_argument("--telegram-test", action="store_true", help="Telegramにテスト送信")
    ap.add_argument("--report", action="store_true", help="仮想売買の成績を表示")
    args = ap.parse_args()

    if args.telegram_test:
        ok = send_telegram("✅ AI株価シグナル：Telegram 接続テスト成功")
        print("送信成功" if ok else "送信できませんでした")
        return 0 if ok else 1

    cfg = load_config()
    con = open_db()
    if args.report:
        print(paper_report(con))
        return 0

    stocks = [s for s in load_watchlist() if args.market == "all" or s.market == args.market.upper()]
    print(f"対象 {len(stocks)} 銘柄")

    results: list[Result] = []
    errors: list[str] = []
    trade_logs: dict[str, str] = {}
    # 1) テクニカル分析（全銘柄）
    for s in stocks:
        try:
            r = analyze(s, fetch_prices(s.ticker, s.market), cfg)
            results.append(r)
            print(f"  {s.ticker:10s} {r.signal:5s} score={r.score:3d} RSI={r.rsi:5.1f} vol={r.vol_ratio:.1f}")
        except Exception as e:  # 1銘柄の失敗で全体を止めない
            errors.append(f"{s.ticker}: {e}")
            print(f"  {s.ticker:10s} ERROR {e}")

    # 2) AI分析（候補銘柄だけ）
    apply_ai(results, cfg)

    # 3) 保存・仮想売買
    for r in results:
        if cfg["show_news"] and r.signal in ("BUY", "SELL") and not r.ai:
            r.headlines = fetch_headlines(r.stock.ticker)
        save_result(con, r)
        save_ai(con, r)
        if cfg["paper_trading"]:
            log = update_paper_trade(con, r, cfg)
            if log:
                trade_logs[r.stock.ticker] = log
    con.commit()

    report = paper_report(con) if cfg["paper_trading"] else ""
    write_result_pages(results, trade_logs, report, errors, cfg)
    msg = build_message(results, errors, list(trade_logs.values()), report, args.market, cfg)
    con.close()

    if msg is None:
        print("通知対象なし")
        return 0
    print("\n" + msg)
    if not args.dry_run:
        send_telegram(msg)
    # 全銘柄失敗ならエラー終了（GitHub Actions で気付けるように）
    return 1 if stocks and not results else 0


if __name__ == "__main__":
    sys.exit(main())
