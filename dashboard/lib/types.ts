/** Mirrors the payload written by `monitoring/publish.py`. */

export interface RegimePanel {
  regime: string; confidence: number; confirmed: boolean; consecutive_bars: number;
  flicker_rate: number; flicker_window: number; flicker_threshold: number;
  is_flickering: boolean; size_multiplier: number; volatility_rank: string | null;
  raw_regime: string | null; n_regimes: number | null; model_age_days: number | null;
  timestamp: string | null;
}

export interface PortfolioPanel {
  equity: number; cash: number; buying_power: number; daily_pnl: number;
  daily_pnl_pct: number; allocation: number; target_allocation: number | null;
  leverage: number; n_positions: number; unrealised_pnl: number;
  peak_equity: number; day_start_equity: number; daily_trades: number;
}

export interface PositionRow {
  symbol: string; direction: string; quantity: number; entry_price: number;
  current_price: number; market_value: number; stop_loss: number | null;
  has_stop: boolean; unrealised_pnl: number; unrealised_pnl_pct: number;
  distance_to_stop_pct: number; regime_at_entry: string; regime_current: string;
  regime_changed: boolean; held_for: string; adopted: boolean;
}

export interface SignalRow {
  timestamp: string; event: string; symbol?: string; shares?: number;
  notional?: number; signal_regime?: string; message?: string;
  rejection_reason?: string | null;
}

export interface RiskPanel {
  halted: boolean; daily_tripped: string; weekly_tripped: string;
  peak_tripped: boolean; size_multiplier: number; n_triggers: number;
  breaker_now?: string;
  drawdowns?: { daily: number; weekly: number; from_peak: number };
  limits: {
    daily_reduce: number; daily_halt: number; weekly_reduce: number;
    weekly_halt: number; max_from_peak: number; max_exposure: number;
    max_leverage: number; max_risk_per_trade: number;
  };
}

export interface SystemPanel {
  data_feed_healthy: boolean; broker_connected: boolean; api_latency_ms: number | null;
  model_age_days: number | null; paper: boolean; mode: string;
  market_open: boolean | null; bars_processed: number; consecutive_errors: number;
  started_at: string | null; symbols: string[]; timeframe: string | null;
}

export interface Candidate {
  symbol: string; rank: number; approved: boolean; action: string; conviction: number;
  shares: number; notional: number; entry_price: number; stop_loss: number | null;
  stop_distance_pct: number; stop_atr_mult: number; risk_dollars: number;
  risk_pct_of_equity: number; trend: string; price_vs_ema50: number; atr_pct: number;
  return_20d: number; strategy: string; regime: string; regime_confidence: number;
  volatility_rank: string; held: boolean; held_quantity: number;
  rejection_reason: string | null; reason: string; modifications: string[]; reasoning: string;
}

export interface Freshness {
  bar_timestamp: string | null; bar_age_hours: number | null; timeframe: string | null;
  sip_delay_minutes: number; publish_cadence: string; poll_seconds: number; realtime: boolean;
}

export interface Timing {
  market_open: boolean | null; next_open: string | null; timeframe: string | null;
  acts_on: string; order_type: string; limit_offset_pct: number; session_close_et: string;
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
  regime_mix: { regime: string; bars: number; pct: number }[];
  candidates: Candidate[];
  freshness: Freshness;
  timing: Timing;
  notes?: Record<string, unknown>;
}
