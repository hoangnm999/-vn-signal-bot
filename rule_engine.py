"""
rule_engine.py — DSL parser + compiler cho /backtest_rule.

Chuyển đổi rule text → pd.Series[+1/0/-1] hoàn toàn vectorized,
KHÔNG dùng eval() trực tiếp — parse qua AST Python để kiểm soát an toàn.

==========================================================================
SUPPORTED SYNTAX
==========================================================================

INDICATORS (dùng trong entry/exit rule):
  close, open, high, low, volume           — giá OHLCV trực tiếp
  sma(N)  / sma20 / sma50                  — Simple Moving Average
  ema(N)  / ema12 / ema26                  — Exponential MA
  rsi(N)  / rsi   / rsi14                  — RSI (default N=14)
  macd()  / macd                           — MACD line (ema12-ema26)
  macd_signal()                            — MACD signal (ema9 of macd)
  macd_hist()                              — MACD histogram
  bb_upper(N,K) / bb_upper                 — Bollinger upper (N=20, K=2)
  bb_lower(N,K) / bb_lower                 — Bollinger lower
  bb_mid(N)    / bb_mid                    — Bollinger mid (=SMA)
  atr(N)  / atr                            — Average True Range (default N=14)
  volume_sma(N) / volume_sma20             — Volume MA
  high_N  / high20  / breakout_high(N)     — N-bar rolling high
  low_N   / low20   / breakout_low(N)      — N-bar rolling low
  stoch_k(N) / stoch_d(N)                  — Stochastic
  prev(indicator, N)                       — giá trị N bar trước của indicator
  NUMBER                                   — literal (vd: 30, 62100, 0.5)

COMPARISONS:
  indicator > value
  indicator < value
  indicator >= value
  indicator <= value
  indicator == value (cũng viết là =)

LOGICAL:
  rule1 and rule2
  rule1 or rule2
  not rule1
  (rule1) and (rule2 or rule3)

SPECIAL FUNCTIONS (exit only, nhưng parser vẫn nhận ở entry):
  crossover(a, b)       — a cắt lên b (a[-1]<b[-1] and a[0]>b[0])
  crossunder(a, b)      — a cắt xuống b
  rising(indicator, N)  — indicator tăng N bar liên tiếp
  falling(indicator, N) — indicator giảm N bar liên tiếp
  breakout(field, N)    — close > high N-bar trước
  breakdown(field, N)   — close < low N-bar trước

EXIT-ONLY FUNCTIONS (tích hợp vào backtest loop):
  trailing_stop(PCT)    — trailing stop PCT% từ đỉnh vị thế
  take_profit(PCT)      — chốt lời khi lãi >= PCT%
  stop_loss(PCT)        — cắt lỗ khi lỗ >= PCT%
  hold(N)               — giữ tối đa N bar

==========================================================================
EXAMPLES
==========================================================================
  entry: "rsi < 30 and close > sma20"
  exit:  "rsi > 70"

  entry: "crossover(ema12, ema26)"
  exit:  "crossunder(ema12, ema26) or trailing_stop(5%)"

  entry: "close > high20 and volume > volume_sma20 * 1.5"
  exit:  "take_profit(10%) or stop_loss(5%)"

  entry: "rsi < 30 and close > sma(20) and macd_hist > 0"
  exit:  "rsi > 70 or close < sma(20)"

  entry: "close == 62100"          (exact price trigger)
  entry: "close < 62100"           (giá về vùng)
"""

from __future__ import annotations

import ast
import re
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# WHITELIST — chỉ cho phép các AST node types này, block tất cả còn lại
# ══════════════════════════════════════════════════════════════════════════════
_ALLOWED_NODE_TYPES = {
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.Compare, ast.Call, ast.Attribute, ast.Name,
    ast.Constant, ast.And, ast.Or, ast.Not,
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.Mult, ast.Div, ast.Add, ast.Sub, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Load,
}

# Tên biến được phép ở top level (không phải calls)
_ALLOWED_NAMES = {
    "close", "open", "high", "low", "volume",
    "rsi", "macd", "atr",
    # sma/ema viết không có () → sẽ resolve thành default period
    "sma", "ema",
    "bb_upper", "bb_lower", "bb_mid",
    "macd_signal", "macd_hist",
    "True", "False",
}

# Shorthand patterns: "sma20" → sma(20), "rsi14" → rsi(14), v.v.
_SHORTHAND_RE = re.compile(
    r'\b(sma|ema|rsi|atr|stoch_k|stoch_d|volume_sma|high|low|bb_upper|bb_lower|bb_mid)'
    r'(\d+)\b'
)

# trailing_stop / take_profit / stop_loss với PCT%
_PCT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%')

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParseError(Exception):
    message: str
    rule_text: str = ""
    hint: str = ""

    def __str__(self):
        parts = [f"Loi parse rule: {self.message}"]
        if self.rule_text:
            parts.append(f"  Rule: {self.rule_text!r}")
        if self.hint:
            parts.append(f"  Goi y: {self.hint}")
        return "\n".join(parts)


@dataclass
class ExitConditions:
    """Các điều kiện exit đặc biệt cần xử lý trong backtest loop."""
    trailing_stop_pct: Optional[float] = None   # % trailing từ đỉnh
    take_profit_pct:   Optional[float] = None   # % chốt lời
    stop_loss_pct:     Optional[float] = None   # % cắt lỗ cứng
    hold_bars:         Optional[int]   = None   # max bars giữ

    def has_dynamic(self) -> bool:
        """Có điều kiện nào cần track giá realtime không."""
        return any([
            self.trailing_stop_pct, self.take_profit_pct,
            self.stop_loss_pct, self.hold_bars
        ])


@dataclass
class CompiledRule:
    """Kết quả compile một rule text."""
    raw:              str
    normalized:       str
    exit_conditions:  ExitConditions = field(default_factory=ExitConditions)
    # Hàm tính indicators cần thiết (set of strings)
    required_indicators: set = field(default_factory=set)


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR LIBRARY — vectorized pandas
# ══════════════════════════════════════════════════════════════════════════════

def _sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()

def _ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def _macd_line(close: pd.Series) -> pd.Series:
    return _ema(close, 12) - _ema(close, 26)

def _macd_signal_line(close: pd.Series) -> pd.Series:
    return _ema(_macd_line(close), 9)

def _macd_hist(close: pd.Series) -> pd.Series:
    ml = _macd_line(close)
    return ml - _ema(ml, 9)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _bb(close: pd.Series, n: int = 20, k: float = 2.0):
    mid   = _sma(close, n)
    std   = close.rolling(n, min_periods=n).std()
    return mid + k * std, mid, mid - k * std   # upper, mid, lower

def _stoch_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    denom  = (high_n - low_n).replace(0, 1e-9)
    return 100 * (df["close"] - low_n) / denom

def _stoch_d(df: pd.DataFrame, n: int = 14, d: int = 3) -> pd.Series:
    return _stoch_k(df, n).rolling(d).mean()

def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """a cắt LÊN b: hôm trước a<b, hôm nay a>b."""
    return ((a > b) & (a.shift(1) < b.shift(1))).astype(int)

def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """a cắt XUỐNG b."""
    return ((a < b) & (a.shift(1) > b.shift(1))).astype(int)

def _rising(s: pd.Series, n: int) -> pd.Series:
    """Tăng n bar liên tiếp."""
    diff = s.diff()
    return (diff > 0).rolling(n).min().fillna(0).astype(int)

def _falling(s: pd.Series, n: int) -> pd.Series:
    diff = s.diff()
    return (diff < 0).rolling(n).min().fillna(0).astype(int)


def build_indicator_context(df: pd.DataFrame) -> dict:
    """
    Tính toàn bộ indicators thường dùng, trả về dict name→Series.
    Gọi một lần duy nhất để tránh tính lại nhiều lần.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    ctx: dict = {
        # OHLCV gốc
        "close":   close,
        "open":    df["open"],
        "high":    high,
        "low":     low,
        "volume":  volume,

        # SMA
        "sma5":    _sma(close, 5),
        "sma10":   _sma(close, 10),
        "sma20":   _sma(close, 20),
        "sma50":   _sma(close, 50),
        "sma100":  _sma(close, 100),
        "sma200":  _sma(close, 200),

        # EMA
        "ema5":    _ema(close, 5),
        "ema9":    _ema(close, 9),
        "ema12":   _ema(close, 12),
        "ema20":   _ema(close, 20),
        "ema26":   _ema(close, 26),
        "ema50":   _ema(close, 50),
        "ema200":  _ema(close, 200),

        # RSI
        "rsi":     _rsi(close, 14),
        "rsi14":   _rsi(close, 14),
        "rsi7":    _rsi(close, 7),

        # MACD
        "macd":         _macd_line(close),
        "macd_signal":  _macd_signal_line(close),
        "macd_hist":    _macd_hist(close),

        # ATR
        "atr":    _atr(df, 14),
        "atr14":  _atr(df, 14),

        # Bollinger Bands (default 20,2)
        "bb_upper": _bb(close, 20, 2)[0],
        "bb_mid":   _bb(close, 20, 2)[1],
        "bb_lower": _bb(close, 20, 2)[2],

        # Volume MA
        "volume_sma5":   _sma(volume, 5),
        "volume_sma10":  _sma(volume, 10),
        "volume_sma20":  _sma(volume, 20),
        "volume_sma":    _sma(volume, 20),

        # Rolling high/low (breakout)
        "high5":   high.rolling(5).max(),
        "high10":  high.rolling(10).max(),
        "high20":  high.rolling(20).max(),
        "high52":  high.rolling(52).max(),
        "low5":    low.rolling(5).min(),
        "low10":   low.rolling(10).min(),
        "low20":   low.rolling(20).min(),
        "low52":   low.rolling(52).min(),

        # Stochastic
        "stoch_k":  _stoch_k(df, 14),
        "stoch_d":  _stoch_d(df, 14),

        # Util aliases
        "sma":  _sma(close, 20),   # unqualified "sma" → sma20
        "ema":  _ema(close, 20),   # unqualified "ema" → ema20
    }
    return ctx


def _resolve_dynamic_indicator(name: str, args: list, df: pd.DataFrame, ctx: dict) -> pd.Series:
    """
    Giải quyết các indicator được gọi với tham số động.
    VD: sma(30), ema(7), rsi(21), bb_upper(20,2), atr(10), ...
    """
    close  = df["close"]
    volume = df["volume"]

    if name == "sma":
        n = int(args[0]) if args else 20
        return _sma(close, n)
    if name == "ema":
        n = int(args[0]) if args else 20
        return _ema(close, n)
    if name == "rsi":
        n = int(args[0]) if args else 14
        return _rsi(close, n)
    if name == "atr":
        n = int(args[0]) if args else 14
        return _atr(df, n)
    if name in ("bb_upper", "bb_lower", "bb_mid"):
        n = int(args[0]) if args else 20
        k = float(args[1]) if len(args) > 1 else 2.0
        u, m, l = _bb(close, n, k)
        return {"bb_upper": u, "bb_lower": l, "bb_mid": m}[name]
    if name == "volume_sma":
        n = int(args[0]) if args else 20
        return _sma(volume, n)
    if name in ("high", "breakout_high"):
        n = int(args[0]) if args else 20
        return df["high"].rolling(n).max()
    if name in ("low", "breakout_low"):
        n = int(args[0]) if args else 20
        return df["low"].rolling(n).min()
    if name == "stoch_k":
        n = int(args[0]) if args else 14
        return _stoch_k(df, n)
    if name == "stoch_d":
        n = int(args[0]) if args else 14
        return _stoch_d(df, n)
    if name == "crossover":
        a = _get_series(args[0], df, ctx)
        b = _get_series(args[1], df, ctx)
        return _crossover(a, b).astype(float)
    if name == "crossunder":
        a = _get_series(args[0], df, ctx)
        b = _get_series(args[1], df, ctx)
        return _crossunder(a, b).astype(float)
    if name == "rising":
        s = _get_series(args[0], df, ctx)
        n = int(args[1]) if len(args) > 1 else 3
        return _rising(s, n).astype(float)
    if name == "falling":
        s = _get_series(args[0], df, ctx)
        n = int(args[1]) if len(args) > 1 else 3
        return _falling(s, n).astype(float)
    if name == "breakout":
        # breakout(high, 20) → close > rolling high 20
        n = int(args[1]) if len(args) > 1 else 20
        return df["high"].rolling(n).max()
    if name == "breakdown":
        n = int(args[1]) if len(args) > 1 else 20
        return df["low"].rolling(n).min()
    if name == "prev":
        s = _get_series(args[0], df, ctx)
        n = int(args[1]) if len(args) > 1 else 1
        return s.shift(n)

    raise ParseError(f"Indicator/function '{name}' khong duoc ho tro", hint=_suggest(name))


def _get_series(arg, df: pd.DataFrame, ctx: dict) -> pd.Series:
    """Chuyển arg (string tên indicator hoặc Series) thành pd.Series."""
    if isinstance(arg, pd.Series):
        return arg
    if isinstance(arg, str):
        if arg in ctx:
            return ctx[arg]
        raise ParseError(f"Indicator '{arg}' khong tim thay", hint=_suggest(arg))
    if isinstance(arg, (int, float)):
        return pd.Series(float(arg), index=df.index)
    raise ParseError(f"Khong the chuyen '{arg}' thanh Series")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR — normalize trước khi parse AST
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_rule(text: str) -> tuple[str, ExitConditions]:
    """
    1. Extract và remove trailing_stop/take_profit/stop_loss/hold (chúng
       không thể biểu diễn bằng boolean Series — xử lý trong loop riêng).
    2. Expand shorthand: sma20 → sma(20), rsi14 → rsi(14), ...
    3. Normalize: == → ==, = → == (single equal), % strip.
    4. Trả về (normalized_text, ExitConditions).
    """
    exit_cond = ExitConditions()
    working   = text.strip()

    # ── Tách exit conditions đặc biệt ─────────────────────────────────────
    # trailing_stop(5%) / trailing_stop(5) / trailing_stop(0.05)
    def _extract_pct_fn(fn_name: str, working: str):
        pat = re.compile(
            r'\b' + fn_name + r'\s*\(\s*'
            r'(\d+(?:\.\d+)?)\s*%?\s*\)',
            re.IGNORECASE,
        )
        m = pat.search(working)
        if m:
            raw_pct = float(m.group(1))
            # Nếu nhập 5 hoặc 5% → 5.0; nếu nhập 0.05 → 5.0 (chuẩn hóa)
            pct = raw_pct if raw_pct > 1 else raw_pct * 100
            working = pat.sub("", working).strip()
            # Dọn "and " hoặc " or " thừa ở đầu/cuối
            working = re.sub(r'^\s*(and|or)\s+', '', working, flags=re.IGNORECASE)
            working = re.sub(r'\s+(and|or)\s*$', '', working, flags=re.IGNORECASE)
            return pct, working
        return None, working

    trailing, working = _extract_pct_fn("trailing_stop", working)
    if trailing is not None:
        exit_cond.trailing_stop_pct = trailing

    tp, working = _extract_pct_fn("take_profit", working)
    if tp is not None:
        exit_cond.take_profit_pct = tp

    sl, working = _extract_pct_fn("stop_loss", working)
    if sl is not None:
        exit_cond.stop_loss_pct = sl

    # hold(N)
    hold_pat = re.compile(r'\bhold\s*\(\s*(\d+)\s*\)', re.IGNORECASE)
    hm = hold_pat.search(working)
    if hm:
        exit_cond.hold_bars = int(hm.group(1))
        working = hold_pat.sub("", working).strip()

    # Dọn sạch and/or thừa sau nhiều lần extract (ở đầu, cuối, và giữa double-or)
    working = re.sub(r'\(\s*\)', '', working)                              # () rỗng
    working = re.sub(r'\bor\s+or\b', 'or', working, flags=re.IGNORECASE)  # or or → or
    working = re.sub(r'\band\s+and\b', 'and', working, flags=re.IGNORECASE)
    working = re.sub(r'^\s*(and|or)\s+', '', working, flags=re.IGNORECASE) # đầu
    working = re.sub(r'\s+(and|or)\s*$', '', working, flags=re.IGNORECASE) # cuối
    working = re.sub(r'\s+(or|and)\s+(or|and)\s+', ' or ', working, flags=re.IGNORECASE)  # giữa kép

    # ── Normalize ─────────────────────────────────────────────────────────
    # sma20 → sma(20), rsi14 → rsi(14), v.v.
    working = _SHORTHAND_RE.sub(lambda m: f"{m.group(1)}({m.group(2)})", working)

    # Dấu = đơn → ==  (nhưng không đụng >=, <=, !=)
    working = re.sub(r'(?<![><!])=(?!=)', '==', working)

    # % trong comparison → chia 100 (VD: "rsi > 30%" → "rsi > 0.30")
    # Nhưng trong take_profit(10%) đã extract rồi nên chỉ xử lý số lẻ
    def _strip_pct(m):
        val = float(m.group(1))
        return str(val / 100 if val > 1 else val)
    working = _PCT_RE.sub(_strip_pct, working)

    # Nếu working rỗng sau khi extract hết special functions
    # → chèn "True" để backtest core biết "luôn exit bằng dynamic condition"
    if not working.replace(" ", ""):
        working = "True"

    return working, exit_cond


# ══════════════════════════════════════════════════════════════════════════════
# AST SECURITY CHECKER
# ══════════════════════════════════════════════════════════════════════════════

def _security_check(tree: ast.AST, raw: str):
    """Kiểm tra AST chỉ chứa node types được whitelist."""
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODE_TYPES:
            raise ParseError(
                f"Phep toan '{type(node).__name__}' khong duoc phep trong rule",
                rule_text=raw,
                hint="Chi dung so sanh, and/or/not, va indicator functions."
            )
        # Block attribute access (df.close, obj.method, ...)
        if isinstance(node, ast.Attribute):
            raise ParseError(
                f"Khong cho phep attribute access ('{node.attr}')",
                rule_text=raw,
                hint="Dung ten indicator truc tiep, vd: 'close' thay vi 'df.close'.",
            )
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            # Nếu là Name node thuần (không phải function call) → block nếu không phải indicator
            # Name nodes trong Call (function names) được check riêng bên dưới
            # Kiểm tra xem node này có phải func của Call không
            pass  # handled below in Call check and at runtime via ctx lookup
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn = node.func.id
                allowed_fns = {
                    "sma", "ema", "rsi", "macd", "atr",
                    "bb_upper", "bb_lower", "bb_mid",
                    "volume_sma", "stoch_k", "stoch_d",
                    "crossover", "crossunder",
                    "rising", "falling",
                    "breakout", "breakdown",
                    "breakout_high", "breakout_low",
                    "high", "low", "prev",
                    "macd_signal", "macd_hist",
                    "trailing_stop", "take_profit", "stop_loss", "hold",
                }
                if fn not in allowed_fns:
                    raise ParseError(
                        f"Function '{fn}' khong duoc ho tro",
                        rule_text=raw,
                        hint=_suggest(fn),
                    )
            elif isinstance(node.func, ast.Attribute):
                raise ParseError(
                    "Khong cho phep attribute access (vd: df.close)",
                    rule_text=raw,
                )


def _suggest(name: str) -> str:
    """Gợi ý gần nhất cho tên sai."""
    known = [
        "close", "open", "high", "low", "volume",
        "sma", "ema", "rsi", "macd", "atr",
        "bb_upper", "bb_lower", "bb_mid",
        "crossover", "crossunder",
        "trailing_stop", "take_profit", "stop_loss",
        "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "volume_sma",
    ]
    from difflib import get_close_matches
    matches = get_close_matches(name.lower(), known, n=1, cutoff=0.5)
    return f"Y ban noi '{matches[0]}'?" if matches else "Xem /backtest_rule help de biet syntax."


# ══════════════════════════════════════════════════════════════════════════════
# AST EVALUATOR — walk AST node → pd.Series boolean
# ══════════════════════════════════════════════════════════════════════════════

class _ASTEvaluator(ast.NodeVisitor):
    """
    Walk AST expression → trả về pd.Series[bool] hoặc pd.Series[float].
    Mọi operation đều vectorized.
    """

    def __init__(self, df: pd.DataFrame, ctx: dict):
        self.df  = df
        self.ctx = ctx

    def eval(self, node) -> pd.Series:
        return self.visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        return pd.Series(float(node.value), index=self.df.index)

    def visit_Name(self, node):
        name = node.id
        if name in self.ctx:
            return self.ctx[name].astype(float)
        if name == "True":
            return pd.Series(1.0, index=self.df.index)
        if name == "False":
            return pd.Series(0.0, index=self.df.index)
        raise ParseError(f"Bien '{name}' khong xac dinh", hint=_suggest(name))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ParseError("Chi ho tro goi ham don gian, khong ho tro method chain")
        fn_name = node.func.id

        # Resolve args — mỗi arg có thể là Series hoặc scalar
        def _resolve_arg(a):
            v = self.visit(a)
            # Nếu là constant Series → trả về float để dùng làm period
            if isinstance(v, pd.Series) and (v == v.iloc[0]).all():
                return float(v.iloc[0])
            return v

        raw_args = [_resolve_arg(a) for a in node.args]

        if fn_name in ("crossover", "crossunder", "rising", "falling",
                       "breakout", "breakdown", "prev",
                       "breakout_high", "breakout_low"):
            # Các hàm cần xử lý args là tên indicator (string)
            str_args = []
            for a_node, a_val in zip(node.args, raw_args):
                if isinstance(a_node, ast.Name):
                    str_args.append(a_node.id)
                elif isinstance(a_val, (int, float)):
                    str_args.append(a_val)
                else:
                    str_args.append(a_val)
            return _resolve_dynamic_indicator(fn_name, str_args, self.df, self.ctx)

        return _resolve_dynamic_indicator(fn_name, raw_args, self.df, self.ctx)

    def visit_BoolOp(self, node):
        # Mỗi operand phải là bool Series trước khi &/| để tránh float bitwise error
        values = [self.visit(v).fillna(0).astype(bool) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = result & v
        else:  # Or
            result = values[0]
            for v in values[1:]:
                result = result | v
        return result.astype(float)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ~operand.astype(bool)
        if isinstance(node.op, ast.USub):
            return -operand
        return operand

    def visit_BinOp(self, node):
        left  = self.visit(node.left)
        right = self.visit(node.right)
        op    = node.op
        if isinstance(op, ast.Add):  return left + right
        if isinstance(op, ast.Sub):  return left - right
        if isinstance(op, ast.Mult): return left * right
        if isinstance(op, ast.Div):  return left / right.replace(0, 1e-9)
        if isinstance(op, ast.Pow):  return left ** right
        raise ParseError(f"Phep tinh '{type(op).__name__}' chua ho tro")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            # Chuẩn hóa cả hai về float để so sánh numeric đúng
            lf = left.astype(float) if isinstance(left, pd.Series) else left
            rf = right.astype(float) if isinstance(right, pd.Series) else right
            if isinstance(op, ast.Gt):    part = lf > rf
            elif isinstance(op, ast.Lt):  part = lf < rf
            elif isinstance(op, ast.GtE): part = lf >= rf
            elif isinstance(op, ast.LtE): part = lf <= rf
            elif isinstance(op, ast.Eq):  part = (lf - rf).abs() < 1e-6
            elif isinstance(op, ast.NotEq): part = (lf - rf).abs() >= 1e-6
            else:
                raise ParseError(f"Phep so sanh '{type(op).__name__}' chua ho tro")
            result = part.astype(float) if result is None else (result.astype(bool) & part).astype(float)
            left   = right
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC PARSER API
# ══════════════════════════════════════════════════════════════════════════════

def parse_rule(rule_text: str) -> tuple[str, ExitConditions]:
    """
    Validate và preprocess rule text.

    Returns:
        (normalized_text, ExitConditions)

    Raises:
        ParseError nếu rule không hợp lệ.
    """
    if not rule_text or not rule_text.strip():
        raise ParseError("Rule khong duoc de trong")

    normalized, exit_cond = _preprocess_rule(rule_text)

    # Parse AST để kiểm tra syntax
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ParseError(
            f"Loi cu phap: {e.msg}",
            rule_text=rule_text,
            hint="Kiem tra lai dau ngoac, toan tu (and/or/not), va ten indicator.",
        )

    _security_check(tree, rule_text)
    return normalized, exit_cond


def compile_signals(
    rule_text: str,
    df: pd.DataFrame,
    ctx: Optional[dict] = None,
) -> pd.Series:
    """
    Compile rule text → pd.Series[+1/0] (True/False vectorized).

    Args:
        rule_text: Đã qua parse_rule() và preprocess
        df:        DataFrame OHLCV
        ctx:       Indicator context (từ build_indicator_context). Nếu None thì tự build.

    Returns:
        Series[float]: 1.0 nếu điều kiện đúng, 0.0 nếu sai, NaN nếu indicator chưa đủ data.
    """
    if ctx is None:
        ctx = build_indicator_context(df)

    try:
        tree  = ast.parse(rule_text, mode="eval")
        ev    = _ASTEvaluator(df, ctx)
        result = ev.eval(tree)
        if isinstance(result, pd.Series):
            return result.fillna(0).astype(float)
        # Scalar (constant expression như "True")
        return pd.Series(float(bool(result)), index=df.index)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Loi khi tinh toan rule: {e}", rule_text=rule_text)


def generate_rule_signals(
    df: pd.DataFrame,
    entry_rule: str,
    exit_rule: str,
) -> tuple[pd.Series, ExitConditions, ExitConditions, dict]:
    """
    Tạo signal Series từ entry + exit rule.

    Logic:
      - Ngày entry_rule=True  & chưa có vị thế → +1 (MUA ngày kế)
      - Ngày exit_rule=True   & đang có vị thế → -1 (BÁN ngày kế)
      - Nếu entry=True và exit=True cùng ngày  → ưu tiên EXIT

    Returns:
        (signals, entry_exit_cond, exit_exit_cond, ctx_dict)
        signals: Series[int] với values +1/0/-1
    """
    ctx = build_indicator_context(df)

    # Parse + validate
    entry_norm, entry_ec = parse_rule(entry_rule)
    exit_norm,  exit_ec  = parse_rule(exit_rule)

    # Compile → boolean Series
    entry_bool = compile_signals(entry_norm, df, ctx).astype(bool)
    exit_bool  = compile_signals(exit_norm,  df, ctx).astype(bool)

    # State machine vectorized (giữ vị thế)
    n          = len(df)
    signals    = pd.Series(0, index=df.index, dtype=int)
    in_pos     = False

    for i in range(n):
        if not in_pos:
            if entry_bool.iloc[i] and not exit_bool.iloc[i]:
                signals.iloc[i] = 1
                in_pos = True
        else:
            if exit_bool.iloc[i]:
                signals.iloc[i] = -1
                in_pos = False

    return signals, entry_ec, exit_ec, ctx


def format_rule_explanation(entry_rule: str, exit_rule: str) -> str:
    """Tạo text giải thích rule cho user đọc hiểu."""
    lines = ["RULE DA PARSE:"]
    lines.append(f"  ENTRY: {entry_rule}")
    lines.append(f"  EXIT : {exit_rule}")

    # Thống kê indicators được dùng
    all_text = entry_rule + " " + exit_rule
    found = []
    for kw in ["rsi", "sma", "ema", "macd", "atr", "bb_upper", "bb_lower",
               "crossover", "crossunder", "stoch", "volume_sma"]:
        if kw in all_text.lower():
            found.append(kw.upper())
    if found:
        lines.append(f"  Indicators: {', '.join(found)}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC EXIT — xử lý trailing stop / take profit trong backtest loop
# ══════════════════════════════════════════════════════════════════════════════

def apply_dynamic_exit(
    df: pd.DataFrame,
    signals: pd.Series,
    exit_ec: ExitConditions,
    config: dict,
) -> dict:
    """
    Phiên bản backtest_core nhận thêm ExitConditions để xử lý
    trailing_stop, take_profit, stop_loss, hold_bars trong loop.

    Thay thế _run_backtest_core khi exit_ec.has_dynamic() == True.
    """
    capital     = float(config["initial_capital"])
    comm_pct    = float(config["commission_pct"])
    slip_pct    = float(config["slippage_pct"])
    pos_size    = float(config.get("position_size", 1.0))

    n          = len(df)
    equity     = np.zeros(n)
    equity[0]  = capital
    cash       = capital
    shares     = 0.0
    entry_price = 0.0
    entry_bar   = -1
    peak_price  = 0.0
    in_pos      = False
    trades      = []

    ts_pct  = (exit_ec.trailing_stop_pct or 0) / 100
    tp_pct  = (exit_ec.take_profit_pct   or 0) / 100
    sl_pct  = (exit_ec.stop_loss_pct     or 0) / 100
    hold_n  = exit_ec.hold_bars or 99999

    for i in range(1, n):
        sig     = int(signals.iloc[i - 1])
        o       = float(df["open"].iloc[i])
        h       = float(df["high"].iloc[i])
        l       = float(df["low"].iloc[i])
        c       = float(df["close"].iloc[i])

        if not in_pos:
            if sig == 1:
                exec_p   = o * (1 + slip_pct)
                invest   = cash * pos_size
                cost_all = exec_p * (1 + comm_pct)
                shares   = invest / cost_all
                cash    -= shares * cost_all
                entry_price = exec_p
                peak_price  = exec_p
                entry_bar   = i
                in_pos      = True
                trades.append({
                    "type":  "BUY",
                    "date":  df["date"].iloc[i].strftime("%Y-%m-%d"),
                    "price": round(exec_p, 0),
                    "shares": round(shares, 4),
                })

        else:
            # Update trailing peak
            peak_price = max(peak_price, h)

            # Kiểm tra dynamic exits (ưu tiên thứ tự: SL → TS → TP → hold → signal)
            do_exit     = False
            exit_reason = ""

            if sl_pct > 0:
                sl_trigger = entry_price * (1 - sl_pct)
                if l <= sl_trigger:
                    do_exit     = True
                    exit_reason = f"StopLoss({exit_ec.stop_loss_pct:.1f}%)"
                    o = max(l, sl_trigger)  # realistic execution near SL

            if not do_exit and ts_pct > 0:
                ts_trigger = peak_price * (1 - ts_pct)
                if l <= ts_trigger:
                    do_exit     = True
                    exit_reason = f"TrailingStop({exit_ec.trailing_stop_pct:.1f}%)"
                    o = max(l, ts_trigger)

            if not do_exit and tp_pct > 0:
                tp_trigger = entry_price * (1 + tp_pct)
                if h >= tp_trigger:
                    do_exit     = True
                    exit_reason = f"TakeProfit({exit_ec.take_profit_pct:.1f}%)"
                    o = min(h, tp_trigger)

            if not do_exit and (i - entry_bar) >= hold_n:
                do_exit     = True
                exit_reason = f"MaxHold({hold_n}bars)"

            if not do_exit and sig == -1:
                do_exit     = True
                exit_reason = "Signal"

            if do_exit:
                exec_p   = o * (1 - slip_pct)
                proceeds = shares * exec_p * (1 - comm_pct)
                cost_in  = shares * entry_price * (1 + comm_pct)
                pnl      = proceeds - cost_in
                pnl_pct  = pnl / cost_in * 100
                cash    += proceeds
                trades.append({
                    "type":      "SELL",
                    "date":      df["date"].iloc[i].strftime("%Y-%m-%d"),
                    "price":     round(exec_p, 0),
                    "shares":    round(shares, 4),
                    "pnl_vnd":   round(pnl, 0),
                    "pnl_pct":   round(pnl_pct, 2),
                    "exit_by":   exit_reason,
                })
                in_pos      = False
                shares      = 0.0
                entry_price = 0.0
                peak_price  = 0.0

        mtm = shares * c if in_pos else 0
        equity[i] = cash + mtm

    if in_pos and shares > 0:
        lc        = float(df["close"].iloc[-1])
        proceeds  = shares * lc * (1 - comm_pct)
        cash     += proceeds
        equity[-1] = cash

    return {"equity": equity, "trades": trades, "final_equity": equity[-1]}
