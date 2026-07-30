"""
QUANTLAB DEMO BOT — Paper Trading
Strategies:
  Family A: BBW_STRICT + RV_LO + DST_NR + PRG_VH   (PF=3.35 n=91)
  Family C: ADX_ST + PBD_HI                          (PF=1.69 n=2049)

This bot NEVER executes live trades.
It monitors, signals, simulates P&L, logs to SQLite, and sends Telegram alerts.

Usage:
    python demo_bot.py               # run live (blocks, fires at :01 each hour)
    python demo_bot.py --scan-now    # run one scan immediately and exit
    python demo_bot.py --status      # print current open positions and equity
    python demo_bot.py --report      # print performance summary and exit

Required env vars (optional — alerts disabled if missing):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import os, sys, time, sqlite3, json, logging, argparse, math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import requests
import numpy as np
import pandas as pd

# ── Optional scheduler ────────────────────────────────────────────────────────
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

# ── Config ────────────────────────────────────────────────────────────────────
DB_FILE        = "demo_bot.db"
LOG_FILE       = "demo_bot.log"
STARTING_CAP   = 10_000.0
RISK_PCT       = 0.01           # 1% per trade
RR             = 2.0
MAX_CONCURRENT = 10             # per strategy
IS_LOOKBACK    = 500            # bars to use as IS window for threshold calibration
RECAL_BARS     = 400            # IS bars needed minimum
MIN_CANDLES    = 250            # minimum history to evaluate a symbol
PAGE_LIMIT     = 100            # OKX candles per page
MAX_PAGES      = 10             # max pages to fetch per symbol (1000 bars)
DAILY_LOSS_LIM = -0.03          # -3% → pause 24h
MAX_DD_LIM     = -0.15          # -15% → halt bot
PAGE_DELAY     = 0.12           # seconds between OKX pages

CACHE_DIR      = "quantlab_cache"          # local parquet cache from research runs

OKX_CANDLES    = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLES_CUR= "https://www.okx.com/api/v5/market/candles"
OKX_INSTR      = "https://www.okx.com/api/v5/public/instruments"
OKX_TICKERS    = "https://www.okx.com/api/v5/market/tickers"

CANDLE_COLS = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("demo_bot")

# =============================================================================
# INDICATORS
# =============================================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_atr(df, period=14):
    h=df["high"]; l=df["low"]; c=df["close"]
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_adx(df, period=14):
    h=df["high"]; l=df["low"]; c=df["close"]
    up=h.diff(); dn=(-l.diff())
    pdm=np.where((up>dn)&(up>0), up, 0.0)
    ndm=np.where((dn>up)&(dn>0), dn, 0.0)
    atr=calc_atr(df, period)
    atr_s=atr.replace(0,np.nan)
    pdi=pd.Series(pdm,index=c.index).ewm(span=period,adjust=False).mean()/atr_s*100
    ndi=pd.Series(ndm,index=c.index).ewm(span=period,adjust=False).mean()/atr_s*100
    dx=(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)*100
    return dx.ewm(span=period,adjust=False).mean()

def add_features(df):
    df = df.copy()
    c=df["close"]; h=df["high"]; l=df["low"]; o=df["open"]
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std(ddof=0)
    df["bb_width"]     = (bb_std*2)/bb_mid.replace(0,np.nan)*100.0
    df["real_vol_20"]  = c.pct_change().rolling(20).std()*100.0
    ema200_safe        = df["ema200"].replace(0,np.nan)
    df["ema_dist_pct"] = (c-ema200_safe)/ema200_safe*100.0
    prev_range         = (h.shift(1)-l.shift(1)).abs()
    prev_body          = (c.shift(1)-o.shift(1)).abs()
    df["prev_range_r"] = prev_range/c.shift(1).replace(0,np.nan)*100.0
    df["prev_body_r"]  = prev_body /c.shift(1).replace(0,np.nan)*100.0
    df["adx14"]        = calc_adx(df, 14)
    df["hour_utc"]     = df.index.hour if hasattr(df.index,"hour") else 0
    vol_avg            = df["vol"].rolling(20).mean()
    df["rel_vol"]      = df["vol"]/vol_avg.replace(0,np.nan)
    return df

# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    "FamilyA": {
        "label":       "Family A",
        "conditions":  ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"],
        "description": "BBW_STRICT+RV_LO+DST_NR+PRG_VH — R066 PF=3.35 n=91",
    },
    "FamilyC": {
        "label":       "Family C",
        "conditions":  ["ADX_ST","PBD_HI"],
        "description": "ADX_ST+PBD_HI — R068 PF=1.69 n=2049",
    },
}

COND_QUANTILE = {
    "BBW_STRICT": ("bb_width",     "lt", 0.25),
    "RV_LO":      ("real_vol_20",  "lt", 0.33),
    "DST_NR":     ("ema_dist_pct", "lt", 0.33),
    "PRG_VH":     ("prev_range_r", "gt", 0.80),
    "ADX_ST":     ("adx14",        "gt", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt", 0.67),
}

def compute_thresholds(df_is, conditions):
    """Compute IS quantile thresholds from in-sample data."""
    thr = {}
    for cid in conditions:
        col, direction, q = COND_QUANTILE[cid]
        vals = df_is[col].dropna()
        if len(vals) < 10:
            thr[cid] = None
            continue
        thr[cid] = float(vals.quantile(q))
    return thr

def check_conditions(bar, thresholds, conditions):
    """Returns (all_ok, detail_dict) for a single bar."""
    detail = {}
    for cid in conditions:
        thr_val = thresholds.get(cid)
        if thr_val is None:
            detail[cid] = False
            continue
        col, direction, _ = COND_QUANTILE[cid]
        val = bar.get(col, np.nan)
        if pd.isna(val):
            detail[cid] = False
            continue
        detail[cid] = (val < thr_val) if direction == "lt" else (val > thr_val)
    return all(detail.values()), detail

def check_entry_gate(bar):
    """RELVOL entry gate: volume spike + green candle + above prev close."""
    return (
        bar.get("rel_vol", 0) > 1.5 and
        bar.get("close", 0)   > bar.get("open",  0) and
        bar.get("close", 0)   > bar.get("close_prev", 0)
    )

# =============================================================================
# OKX DATA FETCHING
# =============================================================================

_last_req = [0.0]
_req_lock_val = 0.05   # 20 req/s

def _throttle():
    wait = _req_lock_val - (time.time() - _last_req[0])
    if wait > 0: time.sleep(wait)
    _last_req[0] = time.time()

def _get(url, params, timeout=15):
    _throttle()
    try:
        r = requests.get(url, params=params, timeout=timeout)
        d = r.json()
        if d.get("code") == "0":
            return d.get("data", [])
    except Exception as e:
        log.warning(f"API error {url}: {e}")
    return []

def fetch_candles(inst_id, n_bars=MIN_CANDLES):
    """Fetch up to n_bars of 1H candles for inst_id. Returns DataFrame or None."""
    all_rows = []; after_ms = None; pages = 0
    while len(all_rows) < n_bars and pages < MAX_PAGES:
        params = {"instId": inst_id, "bar": "1H", "limit": PAGE_LIMIT}
        if after_ms:
            params["after"] = str(after_ms)
        raw = _get(OKX_CANDLES, params)
        if not raw:
            if pages == 0:
                raw = _get(OKX_CANDLES_CUR, params)
            if not raw:
                break
        all_rows.extend(raw); pages += 1
        oldest = int(raw[-1][0])
        after_ms = oldest
        if len(all_rows) >= n_bars:
            break
        time.sleep(PAGE_DELAY)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (df[["datetime","open","high","low","close","vol"]]
          .sort_values("datetime")
          .drop_duplicates("datetime")
          .reset_index(drop=True))
    df = df.set_index("datetime")
    return df if len(df) >= 50 else None

def _build_cache_map():
    """Return dict mapping instId → parquet path for all cached 1H files."""
    cache_map = {}
    cache_path = os.path.join(os.path.dirname(__file__) or ".", CACHE_DIR)
    if not os.path.isdir(cache_path):
        return cache_map
    for fname in os.listdir(cache_path):
        if not fname.endswith("_1H.parquet"):
            continue
        stem    = fname[:-len("_1H.parquet")]          # e.g. BTC_USDT_SWAP
        inst_id = stem.replace("_", "-")               # e.g. BTC-USDT-SWAP
        cache_map[inst_id] = os.path.join(cache_path, fname)
    return cache_map

_CACHE_MAP = None   # populated lazily on first use

def fetch_candles_cached(inst_id, n_bars=None):
    """
    Load candles for inst_id, preferring the local parquet cache.

    Strategy:
      1. If a parquet file exists for inst_id, load it (full history).
      2. Top-up with OKX API for bars newer than the last cached timestamp.
      3. If no cache file exists, fall back to pure OKX API fetch.

    Returns a DataFrame with DatetimeIndex (UTC) and OHLCV columns, or None.
    """
    global _CACHE_MAP
    if _CACHE_MAP is None:
        _CACHE_MAP = _build_cache_map()

    path = _CACHE_MAP.get(inst_id)

    if path and os.path.isfile(path):
        try:
            import pyarrow  # noqa – just ensure it's importable
            df_cache = pd.read_parquet(path)
            df_cache.index = pd.to_datetime(df_cache.index, utc=True)
            df_cache = df_cache.sort_index().drop_duplicates()

            # ── top-up: fetch only the bars we're missing ─────────────────
            last_ts  = df_cache.index[-1]
            now_utc  = pd.Timestamp.now("UTC")
            hours_gap = int((now_utc - last_ts).total_seconds() / 3600)

            if hours_gap >= 2:                          # at least 1 full new bar
                n_topup  = min(hours_gap + 5, PAGE_LIMIT * 3)
                df_new   = fetch_candles(inst_id, n_bars=n_topup)
                if df_new is not None and len(df_new):
                    df_new.index = pd.to_datetime(df_new.index, utc=True)
                    df_combined = pd.concat([df_cache, df_new])
                    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                    df_combined = df_combined.sort_index()
                    log.debug(f"[cache] {inst_id}: {len(df_cache)} cached + "
                              f"{len(df_new)} new → {len(df_combined)} total")
                    return df_combined

            log.debug(f"[cache] {inst_id}: {len(df_cache)} rows from parquet "
                      f"(gap={hours_gap}h, no top-up needed)")
            return df_cache

        except Exception as e:
            log.warning(f"[cache] Failed to load {path}: {e} — falling back to API")

    # ── no cache: pure API ────────────────────────────────────────────────────
    return fetch_candles(inst_id, n_bars=n_bars or (IS_LOOKBACK + MIN_CANDLES + 50))


def fetch_universe():
    """Return list of OKX USDT perp instIds that pass basic volume filter."""
    instr = _get(OKX_INSTR, {"instType": "SWAP"})
    tickers_raw = _get(OKX_TICKERS, {"instType": "SWAP"})
    ticker_map = {t["instId"]: t for t in tickers_raw
                  if t.get("instId","").endswith("-USDT-SWAP")}

    now_ms = int(time.time()*1000)
    min_age_ms = 18 * 30.44 * 24 * 3600 * 1000

    universe = []
    for inst in instr:
        iid = inst.get("instId","")
        if not iid.endswith("-USDT-SWAP"): continue
        if inst.get("state","") != "live": continue
        age = now_ms - int(inst.get("listTime","0") or "0")
        if age < min_age_ms: continue
        tick = ticker_map.get(iid, {})
        vol_usd = float(tick.get("volCcy24h","0") or "0")
        if vol_usd < 5_000_000: continue
        universe.append(iid)

    log.info(f"Universe: {len(universe)} symbols")
    return universe

# =============================================================================
# DATABASE
# =============================================================================

def init_db(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy      TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        entry_time    TEXT NOT NULL,
        entry_price   REAL NOT NULL,
        stop_loss     REAL NOT NULL,
        take_profit   REAL NOT NULL,
        atr           REAL NOT NULL,
        size          REAL NOT NULL,
        risk_usd      REAL NOT NULL,
        exit_time     TEXT,
        exit_price    REAL,
        exit_type     TEXT,
        pnl           REAL,
        status        TEXT DEFAULT 'OPEN'
    );
    CREATE TABLE IF NOT EXISTS signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        conditions    TEXT,
        entry_ok      INTEGER,
        entry_price   REAL,
        atr           REAL
    );
    CREATE TABLE IF NOT EXISTS equity (
        timestamp     TEXT PRIMARY KEY,
        strategy      TEXT NOT NULL,
        capital       REAL NOT NULL,
        peak          REAL NOT NULL,
        drawdown      REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS thresholds (
        updated_at    TEXT NOT NULL,
        strategy      TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        thresholds    TEXT NOT NULL,
        PRIMARY KEY (strategy, symbol)
    );
    """)
    conn.commit()
    return conn

def get_open_trades(conn, strategy=None):
    c = conn.cursor()
    if strategy:
        rows = c.execute("SELECT * FROM trades WHERE status='OPEN' AND strategy=?",
                         (strategy,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()
    cols = [d[0] for d in c.description]
    return [dict(zip(cols,r)) for r in rows]

def open_trade(conn, strategy, symbol, entry_time, entry_price, stop_loss,
               take_profit, atr, size, risk_usd):
    conn.execute("""
        INSERT INTO trades (strategy,symbol,entry_time,entry_price,stop_loss,
                            take_profit,atr,size,risk_usd,status)
        VALUES (?,?,?,?,?,?,?,?,?,'OPEN')
    """, (strategy, symbol, entry_time.isoformat(), entry_price, stop_loss,
          take_profit, atr, size, risk_usd))
    conn.commit()

def close_trade(conn, trade_id, exit_time, exit_price, exit_type, pnl):
    conn.execute("""
        UPDATE trades SET exit_time=?,exit_price=?,exit_type=?,pnl=?,status='CLOSED'
        WHERE id=?
    """, (exit_time.isoformat(), exit_price, exit_type, pnl, trade_id))
    conn.commit()

def log_signal(conn, timestamp, strategy, symbol, conditions, entry_ok,
               entry_price, atr):
    conn.execute("""
        INSERT INTO signals (timestamp,strategy,symbol,conditions,entry_ok,entry_price,atr)
        VALUES (?,?,?,?,?,?,?)
    """, (timestamp.isoformat(), strategy, symbol, json.dumps(conditions),
          int(entry_ok), entry_price, atr))
    conn.commit()

def save_thresholds(conn, strategy, symbol, thr):
    conn.execute("""
        INSERT OR REPLACE INTO thresholds (updated_at,strategy,symbol,thresholds)
        VALUES (?,?,?,?)
    """, (datetime.now(timezone.utc).isoformat(), strategy, symbol,
          json.dumps(thr)))
    conn.commit()

def load_thresholds(conn, strategy, symbol):
    row = conn.execute(
        "SELECT thresholds FROM thresholds WHERE strategy=? AND symbol=?",
        (strategy, symbol)
    ).fetchone()
    return json.loads(row[0]) if row else None

# =============================================================================
# EQUITY / STATE TRACKING
# =============================================================================

class StrategyState:
    """Per-strategy capital and drawdown tracking."""
    def __init__(self, strategy_id, starting_cap=STARTING_CAP):
        self.strategy_id = strategy_id
        self.capital     = starting_cap
        self.peak        = starting_cap
        self.paused_until= None   # datetime or None

    @property
    def drawdown(self):
        return (self.capital - self.peak) / self.peak if self.peak > 0 else 0.0

    def record_pnl(self, pnl):
        self.capital += pnl
        self.peak = max(self.peak, self.capital)

    def is_paused(self):
        if self.paused_until and datetime.now(timezone.utc) < self.paused_until:
            return True
        self.paused_until = None
        return False

    def check_risk_gates(self):
        """Returns (ok, reason). False = do not trade."""
        if self.is_paused():
            return False, "Paused (daily loss limit)"
        if self.drawdown < MAX_DD_LIM:
            return False, f"Max drawdown breach ({self.drawdown:.1%})"
        return True, "OK"

# =============================================================================
# TELEGRAM
# =============================================================================

def _tg_send(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram error: {e}")

def tg_signal(strategy_label, symbol, entry_price, stop_loss, take_profit, atr, rel_vol):
    risk_pct = abs(entry_price - stop_loss) / entry_price * 100
    reward_pct = abs(take_profit - entry_price) / entry_price * 100
    _tg_send(
        f"📡 <b>SIGNAL: {symbol}</b>  [{strategy_label}]\n"
        f"─────────────────────────\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"💰 Entry: ${entry_price:,.4f}\n"
        f"🛑 Stop:  ${stop_loss:,.4f}  (-{risk_pct:.1f}%)\n"
        f"🎯 TP:    ${take_profit:,.4f}  (+{reward_pct:.1f}%)  RR={RR}\n"
        f"📊 ATR: {atr:.4f}  |  RelVol: {rel_vol:.2f}"
    )

def tg_trade_opened(strategy_label, symbol, entry_price, stop_loss,
                    take_profit, size, risk_usd, capital):
    _tg_send(
        f"✅ <b>TRADE OPENED: {symbol}</b>  [{strategy_label}]\n"
        f"─────────────────────────────\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"💵 Entry: ${entry_price:,.4f}\n"
        f"📦 Size: {size:.4f}  |  Risk: ${risk_usd:.2f}\n"
        f"🛑 SL: ${stop_loss:,.4f}  |  🎯 TP: ${take_profit:,.4f}\n"
        f"💼 Capital: ${capital:,.2f}"
    )

def tg_trade_closed(strategy_label, symbol, exit_type, exit_price,
                    pnl, capital, entry_time):
    icon   = "🟢" if pnl > 0 else "🔴"
    result = "WIN" if pnl > 0 else "LOSS"
    hold_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
    _tg_send(
        f"{icon} <b>TRADE CLOSED — {result}: {symbol}</b>  [{strategy_label}]\n"
        f"──────────────────────────────────────\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"✈️ Exit: {exit_type} @ ${exit_price:,.4f}\n"
        f"💰 P&L: {'+' if pnl>=0 else ''}{pnl:.2f} USD\n"
        f"⏱ Hold: {hold_h:.0f}h\n"
        f"📈 Equity: ${capital:,.2f}"
    )

def tg_alert(text):
    _tg_send(f"⚠️ <b>ALERT</b>\n{text}")

def tg_daily_report(states, conn):
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
    lines = [f"📊 <b>DAILY REPORT — {now.strftime('%Y-%m-%d')}</b>\n══════════════════════"]
    for sid, st in states.items():
        label = STRATEGIES[sid]["label"]
        closed = conn.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END),
                   SUM(pnl), SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END)
            FROM trades WHERE strategy=? AND status='CLOSED' AND exit_time >= ?
        """, (sid, day_start)).fetchone()
        n_closed, n_wins, daily_pnl, n_losses = closed
        n_closed  = n_closed  or 0
        n_wins    = n_wins    or 0
        n_losses  = n_losses  or 0
        daily_pnl = daily_pnl or 0.0
        n_open  = len(get_open_trades(conn, sid))
        n_sigs  = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE strategy=? AND timestamp >= ?",
            (sid, day_start)).fetchone()[0]

        all_closed = conn.execute(
            "SELECT pnl FROM trades WHERE strategy=? AND status='CLOSED'", (sid,)
        ).fetchall()
        pnls = [r[0] for r in all_closed if r[0] is not None]
        wins_sum  = sum(p for p in pnls if p > 0)
        loss_sum  = abs(sum(p for p in pnls if p < 0))
        pf = wins_sum/loss_sum if loss_sum > 0 else (999 if wins_sum>0 else 1.0)
        wr = sum(1 for p in pnls if p>0)/len(pnls) if pnls else 0

        lines.append(
            f"\n<b>[{label}]</b>\n"
            f"📡 Signals: {n_sigs}  |  Trades closed: {n_closed}\n"
            f"✅ Wins: {n_wins}  ❌ Losses: {n_losses}\n"
            f"💰 Daily P&L: {'+' if daily_pnl>=0 else ''}{daily_pnl:.2f} USD\n"
            f"📦 Open: {n_open}  |  💵 Equity: ${st.capital:,.2f}\n"
            f"📉 Drawdown: {st.drawdown:.1%}  |  PF all-time: {pf:.3f}  WR: {wr:.1%}"
        )
    _tg_send("\n".join(lines))

# =============================================================================
# CORE SCAN LOGIC
# =============================================================================

def calibrate_symbol(conn, strategy_id, symbol, df):
    """Compute and cache IS thresholds for a symbol."""
    conditions = STRATEGIES[strategy_id]["conditions"]
    df_f = add_features(df)
    df_f.dropna(subset=["ema200","atr14","adx14"], inplace=True)
    if len(df_f) < RECAL_BARS:
        return None
    df_is = df_f.iloc[:IS_LOOKBACK]
    thr = compute_thresholds(df_is, conditions)
    save_thresholds(conn, strategy_id, symbol, thr)
    return thr

def process_symbol(conn, strategy_id, symbol, state, df):
    """
    Evaluate one symbol for one strategy.
    Returns list of actions taken: 'signal', 'trade_opened', 'trade_closed_tp', etc.
    """
    actions = []
    label   = STRATEGIES[strategy_id]["label"]
    conditions = STRATEGIES[strategy_id]["conditions"]
    now     = datetime.now(timezone.utc)

    if len(df) < MIN_CANDLES:
        return actions

    df_f = add_features(df)
    df_f.dropna(subset=["ema200","atr14","adx14"], inplace=True)
    if len(df_f) < 50:
        return actions

    # ── 1. Update open positions: check SL/TP on latest bar ─────────────────
    open_trades = [t for t in get_open_trades(conn, strategy_id)
                   if t["symbol"] == symbol]
    latest_bar = df_f.iloc[-1]  # most recent closed bar

    for trade in open_trades:
        hi  = latest_bar["high"]
        lo  = latest_bar["low"]
        tp  = trade["take_profit"]
        sl  = trade["stop_loss"]
        entry_dt = datetime.fromisoformat(trade["entry_time"])

        if hi >= tp:
            pnl = trade["risk_usd"] * RR
            close_trade(conn, trade["id"], now, tp, "TP", pnl)
            state.record_pnl(pnl)
            log.info(f"[{label}] {symbol} TP hit  pnl=+{pnl:.2f}  equity=${state.capital:.2f}")
            tg_trade_closed(label, symbol, "TP", tp, pnl, state.capital, entry_dt)
            actions.append("trade_closed_tp")

        elif lo <= sl:
            pnl = -trade["risk_usd"]
            close_trade(conn, trade["id"], now, sl, "SL", pnl)
            state.record_pnl(pnl)
            log.info(f"[{label}] {symbol} SL hit  pnl={pnl:.2f}  equity=${state.capital:.2f}")
            tg_trade_closed(label, symbol, "SL", sl, pnl, state.capital, entry_dt)
            actions.append("trade_closed_sl")

    # ── 2. Check risk gates ──────────────────────────────────────────────────
    ok, reason = state.check_risk_gates()
    if not ok:
        log.warning(f"[{label}] Risk gate blocked: {reason}")
        return actions

    # ── 3. Already open on this symbol? ─────────────────────────────────────
    still_open = [t for t in get_open_trades(conn, strategy_id)
                  if t["symbol"] == symbol]
    if still_open:
        return actions

    # ── 4. Max concurrent check ──────────────────────────────────────────────
    total_open = len(get_open_trades(conn, strategy_id))
    if total_open >= MAX_CONCURRENT:
        return actions

    # ── 5. Get/refresh thresholds ────────────────────────────────────────────
    thr = load_thresholds(conn, strategy_id, symbol)
    if thr is None:
        thr = calibrate_symbol(conn, strategy_id, symbol, df)
        if thr is None:
            return actions

    # ── 6. Evaluate signal on the last CLOSED candle (index -2) ──────────────
    if len(df_f) < 3:
        return actions

    sig_bar  = df_f.iloc[-2]
    sig_dict = sig_bar.to_dict()
    sig_dict["close_prev"] = df_f["close"].iloc[-3]

    env_ok, detail = check_conditions(sig_dict, thr, conditions)
    entry_ok = check_entry_gate(sig_dict) if env_ok else False
    atr  = float(sig_bar["atr14"])
    rel_vol = float(sig_dict.get("rel_vol", 0))

    # Always log signal attempts (env_ok only, to reduce noise)
    if env_ok:
        log.info(f"[{label}] {symbol} env_ok={env_ok} entry_ok={entry_ok}  "
                 f"rel_vol={rel_vol:.2f}  atr={atr:.6f}")
        log_signal(conn, now, strategy_id, symbol, detail, entry_ok,
                   float(sig_bar["close"]), atr)

    if not (env_ok and entry_ok):
        return actions

    # ── 7. Open a paper trade ─────────────────────────────────────────────────
    entry_price = float(df_f["close"].iloc[-1])  # current bar = simulated entry
    stop_loss   = entry_price - atr
    take_profit = entry_price + RR * atr
    risk_usd    = state.capital * RISK_PCT
    # position size such that 1 ATR move = risk_usd
    size        = risk_usd / atr if atr > 0 else 0

    if size <= 0 or atr <= 0:
        return actions

    open_trade(conn, strategy_id, symbol, now, entry_price, stop_loss,
               take_profit, atr, size, risk_usd)
    log.info(f"[{label}] TRADE OPENED {symbol}  "
             f"entry={entry_price:.4f}  sl={stop_loss:.4f}  tp={take_profit:.4f}  "
             f"risk=${risk_usd:.2f}")
    tg_signal(label, symbol, entry_price, stop_loss, take_profit, atr, rel_vol)
    tg_trade_opened(label, symbol, entry_price, stop_loss, take_profit,
                    size, risk_usd, state.capital)
    actions.append("trade_opened")
    return actions

# =============================================================================
# HOURLY SCAN
# =============================================================================

def run_scan(conn, states, universe):
    now = datetime.now(timezone.utc)
    log.info(f"{'─'*60}")
    log.info(f"Scan starting  {now.strftime('%Y-%m-%d %H:%M')} UTC  "
             f"symbols={len(universe)}")

    scan_results = defaultdict(lambda: defaultdict(int))

    for inst_id in universe:
        df = fetch_candles_cached(inst_id)
        if df is None or len(df) < MIN_CANDLES:
            continue

        for sid, state in states.items():
            acts = process_symbol(conn, sid, inst_id, state, df)
            for a in acts:
                scan_results[sid][a] += 1

    # Summary log
    for sid, counts in scan_results.items():
        label = STRATEGIES[sid]["label"]
        st = states[sid]
        log.info(
            f"[{label}] scan complete  "
            f"opened={counts.get('trade_opened',0)}  "
            f"tp={counts.get('trade_closed_tp',0)}  "
            f"sl={counts.get('trade_closed_sl',0)}  "
            f"equity=${st.capital:.2f}  dd={st.drawdown:.1%}"
        )

    log.info(f"Scan done  {(datetime.now(timezone.utc)-now).seconds}s")

# =============================================================================
# STATUS / REPORT COMMANDS
# =============================================================================

def print_status(conn, states):
    print(f"\n{'═'*70}")
    print(f"  DEMO BOT STATUS  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'═'*70}\n")

    for sid, st in states.items():
        label = STRATEGIES[sid]["label"]
        open_t = get_open_trades(conn, sid)
        print(f"  [{label}]  equity=${st.capital:,.2f}  dd={st.drawdown:.1%}  "
              f"open_positions={len(open_t)}")
        for t in open_t:
            print(f"    {t['symbol']:<28}  entry={t['entry_price']:.4f}  "
                  f"sl={t['stop_loss']:.4f}  tp={t['take_profit']:.4f}")

def print_report(conn, states):
    print(f"\n{'═'*70}")
    print(f"  PERFORMANCE REPORT  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'═'*70}")

    for sid, st in states.items():
        label = STRATEGIES[sid]["label"]
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE strategy=? AND status='CLOSED'", (sid,)
        ).fetchall()
        pnls = [r[0] for r in rows if r[0] is not None]

        if not pnls:
            print(f"\n  [{label}]  No closed trades yet.")
            continue

        pnls_arr = np.array(pnls)
        wins = pnls_arr[pnls_arr>0]; losses = pnls_arr[pnls_arr<0]
        pf = wins.sum()/abs(losses.sum()) if len(losses)>0 else 999.0
        wr = float((pnls_arr>0).mean())
        eq = np.cumsum(pnls_arr)+STARTING_CAP
        peak = np.maximum.accumulate(eq)
        mdd  = float(((eq-peak)/peak).min())

        print(f"\n  [{label}]  {STRATEGIES[sid]['description']}")
        print(f"    Trades:       {len(pnls_arr)}  (wins={len(wins)}  losses={len(losses)})")
        print(f"    Win rate:     {wr:.1%}")
        print(f"    Profit factor:{pf:.3f}")
        print(f"    Total P&L:    ${pnls_arr.sum():.2f}")
        print(f"    Equity:       ${st.capital:,.2f}")
        print(f"    Max DD:       {mdd:.1%}")
        print(f"    Expectancy:   ${pnls_arr.mean():.2f}/trade")

# =============================================================================
# MAIN
# =============================================================================

def load_states(conn):
    """Load equity state from DB or initialise fresh."""
    states = {}
    for sid in STRATEGIES:
        # Sum all closed trade pnls to reconstruct capital
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE strategy=? AND status='CLOSED'",
            (sid,)
        ).fetchone()
        total_pnl = row[0] if row else 0.0
        st = StrategyState(sid, STARTING_CAP)
        st.capital = STARTING_CAP + total_pnl
        st.peak    = max(STARTING_CAP, st.capital)
        states[sid] = st
    return states

def main():
    parser = argparse.ArgumentParser(description="QuantLab Demo Bot")
    parser.add_argument("--scan-now",  action="store_true", help="Run one scan and exit")
    parser.add_argument("--status",    action="store_true", help="Show open positions and exit")
    parser.add_argument("--report",    action="store_true", help="Show performance report and exit")
    parser.add_argument("--calibrate", action="store_true", help="Recalibrate all thresholds and exit")
    args = parser.parse_args()

    conn   = init_db()
    states = load_states(conn)

    log.info("="*60)
    log.info("QUANTLAB DEMO BOT — starting up")
    for sid, cfg in STRATEGIES.items():
        log.info(f"  {cfg['label']}: {cfg['description']}")
    log.info(f"  DB: {DB_FILE}")
    log.info(f"  Telegram: {'configured' if os.environ.get('TELEGRAM_BOT_TOKEN') else 'NOT configured'}")
    log.info("="*60)

    if args.status:
        print_status(conn, states)
        return

    if args.report:
        print_report(conn, states)
        return

    universe = fetch_universe()
    if not universe:
        log.error("No symbols in universe — check network / OKX API")
        sys.exit(1)

    if args.calibrate:
        log.info("Calibrating thresholds for all symbols …")
        for inst_id in universe:
            df = fetch_candles(inst_id, n_bars=MIN_CANDLES+IS_LOOKBACK)
            if df is None: continue
            for sid in STRATEGIES:
                thr = calibrate_symbol(conn, sid, inst_id, df)
                if thr:
                    log.info(f"  calibrated {inst_id} [{STRATEGIES[sid]['label']}]")
        log.info("Calibration complete.")
        return

    if args.scan_now:
        run_scan(conn, states, universe)
        print_status(conn, states)
        return

    # ── Live mode ─────────────────────────────────────────────────────────────
    if not HAS_SCHEDULER:
        log.error("apscheduler not installed. Run: pip install apscheduler")
        log.error("Or use --scan-now for a single scan.")
        sys.exit(1)

    _universe_cache = {"syms": universe, "fetched_at": datetime.now(timezone.utc)}

    def _refresh_universe():
        _universe_cache["syms"] = fetch_universe()
        _universe_cache["fetched_at"] = datetime.now(timezone.utc)

    def _hourly_job():
        # Refresh universe daily
        age_h = (datetime.now(timezone.utc) - _universe_cache["fetched_at"]).total_seconds()/3600
        if age_h >= 24:
            _refresh_universe()
        run_scan(conn, states, _universe_cache["syms"])

    def _daily_report_job():
        tg_daily_report(states, conn)
        log.info("Daily report sent")

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(_hourly_job,      "cron", minute=1)
    scheduler.add_job(_daily_report_job,"cron", hour=8, minute=0)

    log.info("Scheduler started — scanning at :01 past each hour. Ctrl-C to stop.")
    tg_alert("QuantLab Demo Bot started\n"
             f"Strategies: Family A + Family C\n"
             f"Universe: {len(universe)} symbols")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")

if __name__ == "__main__":
    main()
