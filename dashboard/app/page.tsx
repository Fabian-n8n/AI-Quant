"use client";

import { useEffect, useState } from "react";
import EquityChart from "@/components/EquityChart";
import { Info, Mark, Warning } from "@/components/Icons";
import {
  Positions, PortfolioStats, RegimeHero, RiskStatus, SignalFeed, SystemStatus,
} from "@/components/Panels";
import { relativeTime } from "@/lib/format";
import type { Snapshot } from "@/lib/types";

/** The dashboard reads a published JSON file and nothing else.
 *
 *  It holds no credentials and has no route that can reach a broker, so the
 *  deployed page cannot place an order, clear a breaker, or read the account
 *  directly. A display that can act is no longer a display.
 *
 *  Data arrives via `python main.py --publish`, which writes the same
 *  `DashboardState.snapshot()` the terminal view renders. One source, two
 *  surfaces, no chance of the two disagreeing about what the system thinks. */
export default function Page() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        // Cache-bust: the file is republished in place, and a cached read would
        // show a stale account for as long as the browser felt like it.
        const res = await fetch(`data/state.json?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as Snapshot;
        if (!cancelled) { setSnap(json); setError(null); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "could not load snapshot");
      }
    };

    load();
    // Matches the terminal dashboard's 5-second cadence from the Phase 8 spec.
    const timer = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  if (error && !snap) {
    return (
      <main className="shell">
        <div className="banner banner-neg">
          <Warning />
          <span>
            <strong>No snapshot found.</strong> Run <code>python main.py --publish-demo</code> for
            sample data, or <code>python main.py --dry-run --once --publish</code> to publish your
            own. ({error})
          </span>
        </div>
      </main>
    );
  }

  if (!snap) {
    return (
      <main className="shell">
        <p className="faint">Loading…</p>
      </main>
    );
  }

  const demo = snap.source === "demo";
  const { regime, portfolio, positions, signals, risk, system } = snap;

  return (
    <main className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark"><Mark /></span>
          <div>
            <div className="brand-name">regime-trader</div>
            <div className="brand-sub">
              HMM regime detection · volatility-based allocation
            </div>
          </div>
        </div>

        <div className="badges">
          {demo ? (
            <span className="badge badge-warn"><span className="dot" />Demo data</span>
          ) : (
            <span className="badge badge-accent"><span className="dot" />Live snapshot</span>
          )}
          <span className={`badge ${system.paper ? "badge-pos" : "badge-neg"}`}>
            {system.paper ? "Paper" : "Live money"}
          </span>
          {risk.halted && <span className="badge badge-neg"><span className="dot" />Halted</span>}
          <span className="badge">Updated {relativeTime(snap.published_at)}</span>
        </div>
      </header>

      {demo && (
        <div className="banner banner-warn">
          <Info />
          <span>
            <strong>Demo data, not a real account.</strong> Every figure below is generated
            so the interface can be judged. Publish your own with{" "}
            <code>python main.py --dry-run --once --publish</code>.
          </span>
        </div>
      )}

      {risk.halted && (
        <div className="banner banner-neg">
          <Warning />
          <span>
            <strong>Trading halted by a circuit breaker.</strong> Every signal will be
            rejected until <code>trading_halted.lock</code> is deleted by hand. The friction
            is deliberate: someone should look at what broke before the system can lose more.
          </span>
        </div>
      )}

      <div className="grid grid-main" style={{ marginBottom: "var(--s4)" }}>
        <RegimeHero regime={regime} />
        <PortfolioStats portfolio={portfolio} />
      </div>

      <div className="grid grid-main" style={{ marginBottom: "var(--s4)" }}>
        <section className="card enter" aria-labelledby="equity-heading">
          <div className="card-head">
            <h2 className="card-title" id="equity-heading">Equity and regime history</h2>
            <span className="card-note">{snap.equity_history.length} bars</span>
          </div>
          <EquityChart data={snap.equity_history} regimes={snap.regime_history} />
        </section>
        <RiskStatus risk={risk} />
      </div>

      <div style={{ marginBottom: "var(--s4)" }}>
        <Positions positions={positions} />
      </div>

      <div className="grid grid-halves">
        <SignalFeed signals={signals} />
        <SystemStatus system={system} risk={risk} />
      </div>

      <div className="banner" style={{ marginTop: "var(--s5)", marginBottom: 0 }}>
        <Info />
        <span>
          <strong>This strategy has no demonstrated edge.</strong> Out-of-sample it loses to
          buy-and-hold and to random allocation under identical risk rules, and its drawdown
          trips the peak breaker early in the backtest. The dashboard is honest about what the
          system does; that is not the same as the system working. Paper only.
        </span>
      </div>

      <footer className="footer">
        <span>Read-only view. No credentials, no broker connection, no order path.</span>
        <span className="num">
          Snapshot {new Date(snap.timestamp).toISOString().slice(0, 19).replace("T", " ")} UTC
        </span>
      </footer>
    </main>
  );
}
