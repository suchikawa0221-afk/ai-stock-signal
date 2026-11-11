"""
AI分析（Google Gemini 無料枠）

テクニカル指標の計算は main.py（Python）で行い、AIは
  ・業績（ファンダメンタル）の解釈
  ・ニュース・材料の評価
  ・リスク評価（治験・FDA・決算・規制など）
だけを担当する。AIの判断だけで買いシグナルは出さない。

環境変数 GEMINI_API_KEY が無ければ何もしない（テクニカルのみで動く）。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests
import yfinance as yf

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class AIResult:
    model: str
    fundamental: int  # 0-25
    news: int  # 0-15
    risk: int  # 0-10（高いほど安全）
    stance: str  # 強気 / 中立 / 弱気
    summary: str
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:  # 0-50
        return self.fundamental + self.news + self.risk


def api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def fundamentals(ticker: str) -> dict:
    keys = [
        "longName", "sector", "industry", "marketCap", "currency",
        "trailingPE", "forwardPE", "priceToBook", "profitMargins",
        "revenueGrowth", "earningsGrowth", "totalRevenue", "totalCash", "totalDebt",
        "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions",
    ]
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    return {k: info[k] for k in keys if info.get(k) is not None}


def headlines(ticker: str, limit: int = 8) -> list[str]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items:
        c = it.get("content") or it
        title = c.get("title")
        date = (c.get("pubDate") or "")[:10]
        if title:
            out.append(f"{date} {title}".strip())
        if len(out) >= limit:
            break
    return out


PROMPT = """あなたは慎重な株式アナリストです。次の銘柄を、数日〜数週間のスイングトレードの観点で評価してください。
売買の推奨ではなく、個人投資家が自分で判断するための材料整理が目的です。

# 銘柄
{ticker} {name}（{market}）{sector_note}

# テクニカル（Pythonで計算済み・正確な値）
{technical}

# 財務データ（yfinance）
{fund}

# 最近のニュース見出し
{news}

# 指示
- {search_note}
- 直近の決算内容、売上・利益の成長、業績予想の修正、新製品・提携・規制・訴訟、業界動向を考慮する。
- 治験結果・FDA判断・決算発表など、近い将来の急変動イベントがあればリスクとして必ず挙げる。
- 情報が無い・古い場合は推測せず、点数を中間にして summary にその旨を書く。
- 出力は次のJSONだけ（```json で囲む）。文章はすべて日本語。

```json
{{
  "fundamental": 0〜25の整数（業績・財務の良さ）,
  "news": 0〜15の整数（材料・ニュースの良さ）,
  "risk": 0〜10の整数（10=リスク小、0=急落リスク大）,
  "stance": "強気" か "中立" か "弱気",
  "summary": "80字以内の要約",
  "catalysts": ["上昇材料（各30字以内、最大3つ）"],
  "risks": ["注意点（各30字以内、最大3つ）"]
}}
```"""


def _call(model: str, prompt: str, use_search: bool, timeout: int = 90) -> str:
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    if use_search:
        body["tools"] = [{"google_search": {}}]
    res = requests.post(
        API.format(model=model),
        headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} {res.text[:200]}")
    data = res.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _parse(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        raise ValueError("AIの回答にJSONがありません")
    return json.loads(m.group(1))


def _clamp(v, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return (lo + hi) // 2


def analyze(stock, technical: str, cfg: dict) -> AIResult | None:
    """1銘柄をAIで評価。失敗したら例外。キーが無ければ None。"""
    if not api_key():
        return None
    fund = fundamentals(stock.ticker)
    news = headlines(stock.ticker)
    use_search = bool(cfg.get("ai_web_search", True))
    prompt = PROMPT.format(
        ticker=stock.ticker,
        name=stock.name,
        market="日本株" if stock.market == "JP" else "米国株",
        sector_note="※バイオ株" if stock.sector in ("bio", "biotech", "バイオ") else "",
        technical=technical,
        fund=json.dumps(fund, ensure_ascii=False) if fund else "取得できず",
        news="\n".join(f"- {h}" for h in news) if news else "取得できず",
        search_note="Google検索で直近1か月のニュース・決算・開示を確認してから評価する" if use_search
        else "上の情報だけで評価する",
    )

    last_err: Exception | None = None
    for model in cfg.get("ai_models", ["gemini-flash-latest"]):
        for search in ([True, False] if use_search else [False]):
            try:
                d = _parse(_call(model, prompt, search))
                stance = str(d.get("stance", "中立"))
                return AIResult(
                    model=model + ("+検索" if search else ""),
                    fundamental=_clamp(d.get("fundamental"), 0, 25),
                    news=_clamp(d.get("news"), 0, 15),
                    risk=_clamp(d.get("risk"), 0, 10),
                    stance=stance if stance in ("強気", "中立", "弱気") else "中立",
                    summary=str(d.get("summary", ""))[:200],
                    catalysts=[str(x)[:60] for x in (d.get("catalysts") or [])][:3],
                    risks=[str(x)[:60] for x in (d.get("risks") or [])][:3],
                )
            except Exception as e:  # モデル不在・上限・検索非対応などは次を試す
                last_err = e
                if "429" in str(e):
                    time.sleep(20)
                continue
    raise RuntimeError(f"AI分析に失敗: {last_err}")
