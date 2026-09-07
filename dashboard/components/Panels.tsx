"use client";

import {
  clockTime, humanise, money, price, pct, riskTone, signedMoney, signedPct, toneColor,
} from "@/lib/format";
import type { PortfolioPanel, PositionRow, RegimePanel, RiskPanel, SignalRow, SystemPanel } from "@/lib/types";
import { Empty, Warning } from "./Icons";

/* ---------------------------------------------------------------- regime -- */

/** Confidence as a ring rather than a number alone.
 *
 *  The HMM's probability is the one figure that changes how much every other
 *  number on this page should be trusted, so it gets a shape you can read
 *  without parsing digits. Colour is graded on the config's own 0.55 floor. */
function ConfidenceRing({ value }: { value: number }) {
  const R = 42, C = 2 * Math.PI * R;
  const tone = value >= 0.7 ? "pos" : value >= 0.55 ? "warn" : "neg";
  return (
    <svg width="112" height="112" viewBox="0 0 112 112" role="img"
         aria-label={`Regime confidence ${Math.round(value * 100)} percent`}>
      <circle cx="56" cy="56" r={R} fill="none" stroke="var(--surface-3)" strokeWidth="7" />
      <circle
        cx="56" cy="56" r={R} fill="none" stroke={toneColor(tone)} strokeWidth="7"
        strokeLinecap="round" strokeDasharray={`${C * value} ${C}`}
        transform="rotate(-90 56 56)"
        style={{ transition: "stroke-dasharray 400ms var(--ease)" }}
      />
      <text x="56" y="52" textAnchor="middle" fontSize="24" fontWeight="650"
            fill="var(--fg)" className="num">{Math.round(value * 100)}%</text>
      <text x="56" y="70" textAnchor="middle" fontSize="9.5" fill="var(--fg-faint)"
            letterSpacing="1.1">CONFIDENCE</text>
    </svg>
  );
}

export function RegimeHero({ regime }: { regime: RegimePanel }) {
  const flickerTone = regime.is_flickering ? "neg" : "dim";
  return (
    <section className="card enter" aria-labelledby="regime-heading">
      <div className="card-head">
        <h2 className="card-title" id="regime-heading">Detected regime</h2>
        <span className="card-note">
          {regime.n_regimes ? `${regime.n_regimes} states fitted` : "no model"}
        </span>
      </div>

      <div className="regime">
        <ConfidenceRing value={regime.confidence} />
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--s3)", flexWrap: "wrap" }}>
            <h3 className="regime-name">{humanise(regime.regime)}</h3>
            {regime.confirmed ? (
              <span className="badge badge-accent"><span className="dot" />Confirmed</span>
            ) : (
              <span className="badge badge-warn"><span className="dot" />Pending</span>
            )}
            {regime.is_flickering && (
              <span className="badge badge-neg"><span className="dot" />Flickering</span>
            )}
          </div>

          <div className="regime-meta">
            <div className="meta">
              <div className="meta-label">Stability</div>
              <div className="meta-value num">{regime.consecutive_bars} bars</div>
            </div>
            <div className="meta">
              <div className="meta-label">Flicker</div>
              <div className={`meta-value num ${flickerTone}`}>
                {regime.flicker_rate}/{regime.flicker_window}
              </div>
            </div>
            <div className="meta">
              <div className="meta-label">Volatility rank</div>
              <div className="meta-value">{humanise(regime.volatility_rank)}</div>
            </div>
            <div className="meta">
              <div className="meta-label">Size multiplier</div>
              <div className={`meta-value num ${regime.size_multiplier < 1 ? "warn" : ""}`}>
                {regime.size_multiplier.toFixed(2)}x
              </div>
            </div>
            <div className="meta">
              <div className="meta-label">Model age</div>
              <div className="meta-value num">
                {regime.model_age_days === null ? "—" : `${regime.model_age_days.toFixed(1)}d`}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------- portfolio -- */

export function PortfolioStats({ portfolio }: { portfolio: PortfolioPanel }) {
  const pnlTone = portfolio.daily_pnl >= 0 ? "pos" : "neg";
  const target = portfolio.target_allocation;
  return (
    <section className="card enter" aria-labelledby="portfolio-heading">
      <div className="card-head">
        <h2 className="card-title" id="portfolio-heading">Portfolio</h2>
        <span className="card-note">{portfolio.daily_trades} trades today</span>
      </div>
      <div className="stats">
        <div>
          <div className="stat-label">Equity</div>
          <div className="stat-value num">{money(portfolio.equity)}</div>
          <div className="stat-sub num">peak {money(portfolio.peak_equity)}</div>
        </div>
        <div>
          <div className="stat-label">Daily P&amp;L</div>
          <div className={`stat-value num ${pnlTone}`}>{signedMoney(portfolio.daily_pnl)}</div>
          <div className={`stat-sub num ${pnlTone}`}>{signedPct(portfolio.daily_pnl_pct)}</div>
        </div>
        <div>
          <div className="stat-label">Allocation</div>
          <div className="stat-value num">{pct(portfolio.allocation, 0)}</div>
          <div className="stat-sub num">
            {target === null || target === undefined ? "no target yet" : `target ${pct(target, 0)}`}
          </div>
        </div>
        <div>
          <div className="stat-label">Leverage</div>
          <div className="stat-value num">{portfolio.leverage.toFixed(2)}x</div>
          <div className="stat-sub num">{portfolio.n_positions} positions</div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ risk -- */

function Meter({ name, used, limit, soft }:
  { name: string; used: number; limit: number; soft?: number }) {
  const tone = riskTone(used, limit);
  const ratio = limit ? Math.min(1, Math.abs(used) / Math.abs(limit)) : 0;
  return (
    <div className="meter">
      <div className="meter-head">
        <span className="meter-name">{name}</span>
        <span className={`meter-figure num ${tone}`}>
          {pct(Math.abs(used), 2)} <span className="faint">/ {pct(Math.abs(limit), 0)}</span>
        </span>
      </div>
      <div className="meter-track" role="meter" aria-valuenow={Math.round(ratio * 100)}
           aria-valuemin={0} aria-valuemax={100}
           aria-label={`${name}, ${Math.round(ratio * 100)} percent of limit used`}>
        <div className="meter-fill"
             style={{ width: `${ratio * 100}%`, background: toneColor(tone) }} />
        {soft ? (
          <span className="meter-mark" style={{ left: `${Math.min(100, (soft / limit) * 100)}%` }}
                title={`reduce at ${pct(soft, 0)}`} />
        ) : null}
      </div>
    </div>
  );
}

export function RiskStatus({ risk }: { risk: RiskPanel }) {
  const dd = risk.drawdowns;
  return (
    <section className="card enter" aria-labelledby="risk-heading">
      <div className="card-head">
        <h2 className="card-title" id="risk-heading">Risk status</h2>
        <span className={`badge ${risk.halted ? "badge-neg" : "badge-pos"}`}>
          <span className="dot" />{risk.halted ? "Halted" : "Armed"}
        </span>
      </div>

      {dd ? (
        <>
          <Meter name="Daily drawdown" used={dd.daily} limit={risk.limits.daily_halt}
                 soft={risk.limits.daily_reduce} />
          <Meter name="Weekly drawdown" used={dd.weekly} limit={risk.limits.weekly_halt}
                 soft={risk.limits.weekly_reduce} />
          <Meter name="From peak" used={dd.from_peak} limit={risk.limits.max_from_peak} />
          <p className="card-note" style={{ marginTop: "var(--s4)", lineHeight: 1.55 }}>
            Breakers fire on realised P&amp;L, never on what the model believes. The
            tick marks show where sizing halves; the bar ends where trading stops.
          </p>
        </>
      ) : (
        <div className="empty"><Empty /><span>No portfolio snapshot yet.</span></div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------- positions -- */

export function Positions({ positions }: { positions: PositionRow[] }) {
  const naked = positions.filter((p) => !p.has_stop);
  return (
    <section className="card enter" aria-labelledby="positions-heading">
      <div className="card-head">
        <h2 className="card-title" id="positions-heading">Positions</h2>
        <span className="card-note">{positions.length} open</span>
      </div>

      {naked.length > 0 && (
        <div className="banner banner-neg">
          <Warning />
          <span>
            <strong>{naked.map((p) => p.symbol).join(", ")}</strong> {naked.length === 1 ? "has" : "have"}
            {" "}no protective stop. Place one by hand or close the position.
          </span>
        </div>
      )}

      {positions.length === 0 ? (
        <div className="empty"><Empty /><span>No open positions.</span></div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Symbol</th><th scope="col">Qty</th><th scope="col">Entry</th>
                <th scope="col">Last</th><th scope="col">Stop</th><th scope="col">Value</th>
                <th scope="col">P&amp;L</th><th scope="col">Held</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.symbol}>
                  <td>
                    <span className="sym">{p.symbol}</span>
                    {p.regime_changed && (
                      <span className="badge badge-warn" style={{ marginLeft: 8 }}>regime moved</span>
                    )}
                    {p.adopted && (
                      <span className="badge" style={{ marginLeft: 8 }}>adopted</span>
                    )}
                  </td>
                  <td className="num dim">{p.quantity}</td>
                  <td className="num dim">{price(p.entry_price)}</td>
                  <td className="num">{price(p.current_price)}</td>
                  <td className="num">
                    {p.has_stop && p.stop_loss !== null
                      ? <>{price(p.stop_loss)} <span className="faint">
                          ({pct(p.distance_to_stop_pct, 1)})</span></>
                      : <span className="neg">none</span>}
                  </td>
                  <td className="num dim">{money(p.market_value)}</td>
                  <td className={`num ${p.unrealised_pnl >= 0 ? "pos" : "neg"}`}>
                    {signedMoney(p.unrealised_pnl)}{" "}
                    <span style={{ opacity: 0.75 }}>{signedPct(p.unrealised_pnl_pct, 1)}</span>
                  </td>
                  <td className="num faint">{p.held_for}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- signals -- */

export function SignalFeed({ signals }: { signals: SignalRow[] }) {
  const rows = [...signals].reverse().slice(0, 12);
  return (
    <section className="card enter" aria-labelledby="signals-heading">
      <div className="card-head">
        <h2 className="card-title" id="signals-heading">Recent signals</h2>
        <span className="card-note">rejections included</span>
      </div>

      {rows.length === 0 ? (
        <div className="empty"><Empty /><span>No signals yet.</span></div>
      ) : (
        <div className="feed">
          {rows.map((s, i) => {
            const rejected = s.event === "signal_rejected";
            return (
              <div className="feed-row" key={`${s.timestamp}-${s.symbol}-${i}`}>
                <span className="feed-time num">{clockTime(s.timestamp)}</span>
                <span className="sym">{s.symbol ?? "—"}</span>
                <span className="feed-text" title={s.message}>
                  {rejected
                    ? <>rejected <span className="faint">
                        {humanise(s.rejection_reason ?? "")}</span></>
                    : <>{s.shares ?? 0} shares
                        <span className="faint"> · {money(s.notional ?? 0)}</span></>}
                </span>
                <span className={`badge ${rejected ? "badge-neg" : "badge-pos"}`}>
                  {rejected ? "Rejected" : "Approved"}
                </span>
              </div>
            );
          })}
        </div>
      )}
      <p className="card-note" style={{ marginTop: "var(--s4)", lineHeight: 1.55 }}>
        Rejections are shown as prominently as approvals. A feed of only the trades
        that happened cannot tell you the system stopped trading three weeks ago.
      </p>
    </section>
  );
}

/* ----------------------------------------------------------------- system -- */

export function SystemStatus({ system, risk }: { system: SystemPanel; risk: RiskPanel }) {
  const items: { label: string; value: string; tone: string }[] = [
    { label: "Data feed", value: system.data_feed_healthy ? "Healthy" : "Down",
      tone: system.data_feed_healthy ? "pos" : "neg" },
    { label: "Broker API",
      value: system.broker_connected
        ? `Connected${system.api_latency_ms ? ` · ${Math.round(system.api_latency_ms)}ms` : ""}`
        : "Disconnected",
      tone: system.broker_connected ? "pos" : "neg" },
    { label: "Model",
      value: system.model_age_days === null ? "None" : `${system.model_age_days.toFixed(1)}d old`,
      tone: (system.model_age_days ?? 0) > 7 ? "warn" : "pos" },
    { label: "Market", value: system.market_open ? "Open" : "Closed",
      tone: system.market_open ? "pos" : "dim" },
    { label: "Bars processed", value: String(system.bars_processed), tone: "dim" },
    { label: "Breaker", value: humanise(risk.breaker_now ?? "none"), tone:
      risk.breaker_now && risk.breaker_now !== "none" ? "warn" : "pos" },
  ];

  return (
    <section className="card enter" aria-labelledby="system-heading">
      <div className="card-head">
        <h2 className="card-title" id="system-heading">System</h2>
        <span className="card-note">
          {system.symbols.length} symbols · {system.timeframe ?? "—"}
        </span>
      </div>
      <div className="stats">
        {items.map((i) => (
          <div key={i.label}>
            <div className="stat-label">{i.label}</div>
            <div className="stat-value" style={{ fontSize: 16, color: toneColor(i.tone) }}>
              {i.value}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
