/** Mirrors the payload written by `monitoring/publish.py`. */

export interface RegimePanel {
  regime: string;
  confidence: number;
  confirmed: boolean;
  consecutive_bars: number;
  flicker_rate: number;
  flicker_window: number;
  flicker_threshold: number;
  is_flickering: boolean;
  size_multiplier: number;
  volatility_rank: string | null;
  raw_regime: string | null;
  n_regimes: number | null;
  model_age_days: number | null;
  timestamp: string | null;
}

export interface PortfolioPanel {
  equity: number;
  cash: number;
  buying_power: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  allocation: number;
  target_allocation: number | null;
  leverage: number;
  n_positions: number;
  unrealised_pnl: number;
  peak_equity: number;
  day_start_equity: number;
  daily_trades: number;
}

export interface PositionRow {
  symbol: string;
  direction: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  stop_loss: number | null;
  has_stop: boolean;
  unrealised_pnl: number;
  unrealised_pnl_pct: number;
  distance_to_stop_pct: number;
  regime_at_entry: string;
  regime_current: string;
  regime_changed: boolean;
  held_for: string;
  adopted: boolean;
}

export interface SignalRow {
  timestamp: string;
  event: string;
  symbol?: string;
  shares?: number;
  notional?: number;
  signal_regime?: string;
  message?: string;
  rejection_reason?: string | null;
}

export interface RiskPanel {
  halted: boolean;
  daily_tripped: string;
  weekly_tripped: string;
  peak_tripped: boolean;
  size_multiplier: number;
  n_triggers: number;
  breaker_now?: string;
  drawdowns?: { daily: number; weekly: number; from_peak: number };
  limits: {
    daily_reduce: number; daily_halt: number;
    weekly_reduce: number; weekly_halt: number;
    max_from_peak: number; max_exposure: number;
    max_leverage: number; max_risk_per_trade: number;
  };
}

export interface SystemPanel {
  data_feed_healthy: boolean;
  broker_connected: boolean;
  api_latency_ms: number | null;
  model_age_days: number | null;
  paper: boolean;
  mode: string;
  market_open: boolean | null;
  bars_processed: number;
  consecutive_errors: number;
  started_at: string | null;
  symbols: string[];
  timeframe: string | null;
}

export interface Snapshot {
  schema_version: number;
  source: "live" | "demo";
  published_at: string;
  timestamp: string;
  regime: RegimePanel;
  portfolio: PortfolioPanel;
  positions: PositionRow[];
  signals: SignalRow[];
  risk: RiskPanel;
  system: SystemPanel;
  equity_history: { t: string; equity: number; peak: number }[];
  regime_history: { t: string; regime: string }[];
  notes?: Record<string, unknown>;
}
