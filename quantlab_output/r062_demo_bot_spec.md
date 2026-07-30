
# QUANTLAB DEMO BOT — Paper Trading Specification
# Strategy: BBW_STRICT+RV_LO+DST_NR+PRG_VH  (E3.1_v2)
# Generated: R062 | Universe: 46 symbols
# Status: READY FOR PAPER TRADING

## Overview

This specification defines a fully automated paper-trading demo bot that:
- Monitors all OKX USDT perpetual futures in real time (1H candles)
- Detects signals for the frozen E3.1_v2 strategy
- Simulates trades with realistic costs (no live execution)
- Sends Telegram alerts for every signal and trade event
- Publishes a daily performance report

**The bot NEVER executes live trades. It only monitors, signals, and simulates.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DEMO BOT — COMPONENTS                                          │
│                                                                 │
│  [Scheduler] ──→ [Market Scanner] ──→ [Signal Engine]          │
│                        │                     │                 │
│                  [OKX REST API]       [Position Tracker]        │
│                                              │                 │
│                                    [Trade Log (SQLite)]         │
│                                              │                 │
│                               [Telegram Alerter] ←→ [Reporter] │
└─────────────────────────────────────────────────────────────────┘
```

**Technology stack:**
- Language: Python 3.11+
- Scheduler: `apscheduler` (cron-based, fires at :01 past each hour)
- Database: SQLite (local, single file)
- Alerting: `python-telegram-bot`
- Data: OKX public REST API (no API key required for candles)
- Config: YAML file (`bot_config.yaml`)

---

## Configuration File (`bot_config.yaml`)

```yaml
# ── Universe ────────────────────────────────────────────────────
universe:
  min_vol_24h_usd: 5_000_000      # minimum 24h volume to monitor
  min_history_months: 18          # skip if listed < 18 months ago
  update_every_hours: 24          # re-scan universe daily

# ── Frozen strategy parameters — DO NOT CHANGE ──────────────────
strategy:
  conditions:
    - BBW_STRICT                  # bb_width < IS 25th percentile
    - RV_LO                       # realised_vol_20 < IS 33rd percentile
    - DST_NR                      # ema_dist_pct < IS 33rd percentile
    - PRG_VH                      # prev_range_r > IS 80th percentile
  entry:
    rel_vol_threshold: 1.5        # RELVOL > 1.5
    require_bullish_candle: true  # close > open
    require_above_prev_close: true # close > prev_close
  exit:
    risk_reward: 2.0              # take-profit = entry + 2×ATR
    stop_loss: entry - 1×ATR      # stop = entry - 1×ATR (prev ATR14)

# ── Position sizing (paper only) ────────────────────────────────
position_sizing:
  starting_capital: 10000.0      # simulated capital USD
  risk_per_trade_pct: 0.01       # 1% risk per trade
  max_leverage: 5.0              # position cap
  max_concurrent_positions: 10   # max open simulated positions

# ── Costs (realistic simulation) ────────────────────────────────
costs:
  taker_fee: 0.0005
  spread: 0.0002
  sl_slippage: 0.0003
  min_sl_pct: 0.001

# ── IS window for threshold calibration ─────────────────────────
thresholds:
  is_lookback_months: 18         # use last 18M of history as IS data
  recalibrate_every_days: 7      # refresh thresholds weekly

# ── Telegram ────────────────────────────────────────────────────
telegram:
  bot_token: "SET_IN_ENV_VAR"    # TELEGRAM_BOT_TOKEN env var
  chat_id:   "SET_IN_ENV_VAR"    # TELEGRAM_CHAT_ID env var
  signal_alerts: true
  entry_alerts: true
  exit_alerts: true
  daily_report: true
  daily_report_time: "08:00"     # UTC

# ── Logging ─────────────────────────────────────────────────────
logging:
  db_file: "demo_bot.db"
  log_file: "demo_bot.log"
  log_level: INFO
```

---

## Entry Logic

```python
def check_entry_signal(df_candles: pd.DataFrame, thresholds: dict) -> bool:
    """
    Called on the CLOSED candle at each :00 UTC hour.
    Uses the previous candle (index -2) as the signal candle;
    entry simulation occurs on the NEXT open (index -1).
    """
    df = add_features(df_candles)          # computes all indicators
    sig_bar = df.iloc[-2]                  # last fully closed candle

    # Environment conditions (all must be True)
    env_ok = all([
        sig_bar["bb_width"]     < thresholds["BBW_STRICT"],   # compression
        sig_bar["real_vol_20"]  < thresholds["RV_LO"],        # low realised vol
        sig_bar["ema_dist_pct"] < thresholds["DST_NR"],       # near EMA200
        sig_bar["prev_range_r"] > thresholds["PRG_VH"],       # high prev range
    ])
    if not env_ok:
        return False

    # Entry conditions
    relvol  = sig_bar["rel_vol"] > 1.5           # volume spike
    bullish = sig_bar["close"]   > sig_bar["open"]  # green candle
    above   = sig_bar["close"]   > sig_bar["prev_close"]  # above prev close

    return relvol and bullish and above
```

---

## Exit Logic

```python
def compute_exit_levels(entry_price: float, atr: float,
                        rr: float = 2.0) -> tuple:
    """
    Returns (stop_loss, take_profit) for a LONG paper trade.
    ATR = prev_atr14 at time of entry bar.
    """
    stop_loss   = entry_price - atr
    take_profit = entry_price + rr * atr
    return stop_loss, take_profit
```

---

## Position Sizing

```python
def calc_position_size(capital: float, atr: float,
                       entry_price: float,
                       risk_pct: float = 0.01,
                       max_lev: float = 5.0) -> float:
    """
    Returns notional size (number of contracts at entry_price USD each).
    Risk $100 (1% of $10k) on the ATR stop distance.
    Capped at max_lev × capital.
    """
    risk_dollars = capital * risk_pct
    size = min(risk_dollars / atr, (capital * max_lev) / entry_price)
    return max(size, 0.0)
```

---

## Risk Management

| Rule | Value |
|------|-------|
| Max concurrent positions | 10 |
| Risk per trade | 1% of equity |
| Max leverage | 5× |
| Session filter | None (24/7) |
| Daily loss limit | −3% of capital → pause for 24h |
| Max equity drawdown | −15% → halt and alert |

```python
def check_risk_gates(state: dict) -> bool:
    """Returns False if daily loss limit or max DD limit is breached."""
    daily_pnl_pct = state["daily_pnl"] / state["capital"]
    max_dd_pct    = state["max_drawdown"]
    if daily_pnl_pct < -0.03:
        send_telegram("⚠️ Daily loss limit reached (−3%). Pausing 24h.")
        return False
    if max_dd_pct < -0.15:
        send_telegram("🚨 Max drawdown breach (−15%). Bot halted. Manual review needed.")
        return False
    return True
```

---

## Trade Logging (SQLite schema)

```sql
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    DATETIME NOT NULL,
    symbol       TEXT NOT NULL,
    signal_bar   DATETIME NOT NULL,
    env_ok       BOOLEAN,
    entry_ok     BOOLEAN,
    bb_width     REAL,
    real_vol_20  REAL,
    ema_dist_pct REAL,
    prev_range_r REAL,
    rel_vol      REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    entry_time   DATETIME NOT NULL,
    entry_price  REAL NOT NULL,
    stop_loss    REAL NOT NULL,
    take_profit  REAL NOT NULL,
    atr          REAL NOT NULL,
    size         REAL NOT NULL,
    exit_time    DATETIME,
    exit_price   REAL,
    exit_type    TEXT,  -- 'TP' | 'SL' | 'MANUAL'
    pnl          REAL,
    pnl_pct      REAL,
    fees         REAL,
    status       TEXT DEFAULT 'OPEN'  -- 'OPEN' | 'CLOSED'
);

CREATE TABLE IF NOT EXISTS daily_reports (
    date         TEXT PRIMARY KEY,
    n_signals    INTEGER,
    n_trades     INTEGER,
    n_open       INTEGER,
    n_closed     INTEGER,
    n_wins       INTEGER,
    n_losses     INTEGER,
    daily_pnl    REAL,
    equity       REAL,
    peak_equity  REAL,
    drawdown     REAL
);
```

---

## Telegram Alerts

**Signal detected:**
```
📡 SIGNAL: ETH-USDT-SWAP
─────────────────────────
📅 2024-11-15 14:00 UTC
💰 Entry: $2,847.50
🛑 Stop:  $2,798.30  (−1.7%)
🎯 TP:    $2,945.90  (+3.4%)  RR=2.0
📊 ATR: $49.20 | RelVol: 2.14
📉 BBW: 0.031 | RV: 0.48%
```

**Trade opened:**
```
✅ TRADE OPENED: SOL-USDT-SWAP
─────────────────────────────
📅 2024-11-15 14:00 UTC
💵 Entry: $187.42
📦 Size: 5.32 contracts ($997 notional)
🛑 SL: $183.10 | 🎯 TP: $195.98
💼 Capital at risk: $23.02 (1.0%)
```

**Trade closed (TP hit):**
```
🟢 TRADE CLOSED — WIN: BTC-USDT-SWAP
──────────────────────────────────────
📅 2024-11-16 02:00 UTC
✈️ Exit: TP hit @ $68,420
💰 P&L: +$89.40 (+0.9%)
⏱  Hold: 12h
📈 Equity: $10,284 (+2.8% all-time)
```

**Trade closed (SL hit):**
```
🔴 TRADE CLOSED — LOSS: AVAX-USDT-SWAP
───────────────────────────────────────
📅 2024-11-16 05:00 UTC
🛑 Exit: SL hit @ $34.21
💸 P&L: −$43.20 (−0.4%)
⏱  Hold: 8h
📉 Equity: $10,241
```

**Daily report (08:00 UTC):**
```
📊 DAILY REPORT — 2024-11-16
══════════════════════════
📡 Signals today    : 7
📈 Trades opened    : 3
✅ Trades closed TP : 2
❌ Trades closed SL : 1
💰 Daily P&L        : +$135.60 (+1.4%)
─────────────────────────
📦 Open positions   : 2
💵 Equity           : $10,420
📉 Max drawdown     : −2.1%
🏆 Win rate (all)   : 64.3%
📊 Profit factor    : 1.78
```

---

## Equity Curve Tracking

The bot maintains a running equity curve updated after every closed trade:

```python
class EquityTracker:
    def __init__(self, starting_capital: float):
        self.capital      = starting_capital
        self.peak         = starting_capital
        self.trade_log    = []       # list of (timestamp, pnl, equity)
        self.max_drawdown = 0.0

    def record_trade(self, pnl: float, timestamp):
        self.capital += pnl
        self.peak     = max(self.peak, self.capital)
        dd = (self.capital - self.peak) / self.peak
        self.max_drawdown = min(self.max_drawdown, dd)
        self.trade_log.append((timestamp, pnl, self.capital, dd))

    def daily_summary(self):
        # Returns dict of daily P&L, equity, drawdown
        ...
```

---

## Win/Loss Statistics

Updated in real time and included in the daily report:

```python
class Statistics:
    def update(self, trade):
        # Running totals: n_trades, n_wins, n_losses
        # Gross wins / gross losses → profit_factor
        # Running win_rate, avg_win, avg_loss
        # R-multiple (win / avg_loss_abs)
        # Expectancy = win_rate × RR − (1 − win_rate)
        ...
```

---

## Deployment Architecture

### Local (development/testing)

```bash
# Install
pip install apscheduler python-telegram-bot requests pandas pyarrow pyyaml

# Configure
cp bot_config.yaml.template bot_config.yaml
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Run
python demo_bot.py --config bot_config.yaml
```

### VPS / Cloud (production paper trading)

```
Recommended: Ubuntu 22.04 VPS (2 vCPU, 2 GB RAM)
Monthly cost: ~$6/month (DigitalOcean, Vultr, Hetzner)

Deployment:
  1. Clone repository to VPS
  2. Set environment variables (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
  3. Run: systemctl enable demo_bot && systemctl start demo_bot
  4. Monitor: journalctl -f -u demo_bot

Systemd unit file: /etc/systemd/system/demo_bot.service
```

---

## Main Loop (simplified)

```python
# demo_bot.py — main entry point

def run_hourly_scan():
    """Called at :01 past every hour."""
    if not check_risk_gates(state): return

    # 1. Refresh universe (daily)
    if should_refresh_universe():
        universe = fetch_okx_universe()

    # 2. Refresh thresholds (weekly)
    if should_recalibrate():
        for sym in universe:
            df_hist = fetch_history(sym, months=18)
            df_is   = df_hist.iloc[:int(len(df_hist)*0.80)]
            thresholds[sym] = learn_thresholds(add_features(df_is))

    # 3. Scan each symbol
    for sym in universe:
        df = fetch_last_candles(sym, n=300)  # enough for indicators
        if check_entry_signal(df, thresholds[sym]):
            entry_price = df["close"].iloc[-1]
            atr         = df["atr14"].iloc[-2]
            sl, tp      = compute_exit_levels(entry_price, atr)
            size        = calc_position_size(state["capital"], atr, entry_price)
            trade       = open_paper_trade(sym, entry_price, sl, tp, size)
            send_telegram_signal(sym, trade)

    # 4. Update open positions (check SL/TP)
    for trade in get_open_trades():
        current_candle = fetch_last_candle(trade.symbol)
        if current_candle["high"] >= trade.take_profit:
            close_trade(trade, trade.take_profit, "TP")
        elif current_candle["low"] <= trade.stop_loss:
            close_trade(trade, trade.stop_loss * (1 - 0.0003), "SL")

scheduler = BlockingScheduler()
scheduler.add_job(run_hourly_scan, 'cron', minute=1)
scheduler.start()
```

---

## Key Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| No live trading | Bot holds no API write keys; REST calls are GET-only |
| Reproducible signals | All thresholds stored in DB at calibration time |
| Audit trail | Every signal bar and all conditions logged to SQLite |
| Fail-safe | Any API error → skip symbol, log, continue |
| Restart-safe | Open positions recovered from SQLite on restart |
| Transparent costs | All fees, spread, slippage applied identically to R062 backtest |

---

## Paper Trading Calendar

| Phase | Duration | Purpose |
|-------|----------|---------|
| Phase 1: Calibration | Weeks 1-2 | Verify signals match backtest (manual comparison) |
| Phase 2: Observation | Months 1-3 | Accumulate 50+ paper trades |
| Phase 3: Statistics | Month 4 | Compare live paper PF to R062 forward PF |
| Phase 4: Decision | Month 5 | Decide whether to proceed to live allocation |

**Minimum confidence threshold before live trading:**
- 100+ paper trades accumulated
- Paper PF ≥ 1.20 (statistically consistent with R062)
- No evidence of PF degradation across months

