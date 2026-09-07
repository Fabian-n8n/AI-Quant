"use client";

import * as React from "react";
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, Ban, CircleDot,
  Gauge, Info, Layers, ShieldCheck, Signal, Wifi, WifiOff,
} from "lucide-react";
import { AllocationBar, ConfidenceGauge, EquityChart, RegimeMix, Sparkline } from "@/components/charts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Stat } from "@/components/ui/stat";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  clockTime, humanise, money, pct, pnlTone, price, riskTone, signedMoney,
  signedPct, toneText, type Tone,
} from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Snapshot } from "@/lib/types";

/* ------------------------------------------------------------- empty -- */

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-muted-foreground">
      <Layers className="h-6 w-6 opacity-40" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

/* ------------------------------------------------------------ regime -- */

export function RegimeCard({ regime }: { regime: Snapshot["regime"] }) {
  const vol = regime.volatility_rank;
  const volTone: Tone =
    vol === "low" ? "positive" : vol === "high" ? "negative" : vol === "mid" ? "warning" : "muted";

  return (
    <Card span="2">
      <CardHeader>
        <CardTitle>Detected regime</CardTitle>
        <span className="text-xs text-muted-foreground">
          {regime.n_regimes ? `${regime.n_regimes} states fitted` : "no model"}
        </span>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-6">
        <ConfidenceGauge value={regime.confidence} />

        <div className="min-w-0 flex-1 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              {humanise(regime.regime)}
            </h2>
            {regime.confirmed ? (
              <Badge variant="primary" dot pulse>Confirmed</Badge>
            ) : (
              <Badge variant="warning" dot pulse>Pending</Badge>
            )}
            {regime.is_flickering && <Badge variant="negative" dot>Flickering</Badge>}
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Stability" size="sm" value={`${regime.consecutive_bars} bars`} />
            <Stat
              label="Flicker" size="sm"
              tone={regime.is_flickering ? "negative" : "muted"}
              value={`${regime.flicker_rate}/${regime.flicker_window}`}
              sub={`limit ${regime.flicker_threshold}`}
            />
            <Stat label="Volatility" size="sm" tone={volTone} value={humanise(vol)} />
            <Stat
              label="Size" size="sm"
              tone={regime.size_multiplier < 1 ? "warning" : "muted"}
              value={`${regime.size_multiplier.toFixed(2)}x`}
            />
          </div>
        </div>
      </CardContent>
      <CardFooter>
        Allocation follows <strong className="font-medium text-foreground">measured volatility</strong>,
        never the regime label. The two orderings disagree: a state labelled bearish can be the
        calmest one fitted, and calmness is what decides how much capital is at risk.
      </CardFooter>
    </Card>
  );
}

/* --------------------------------------------------------- portfolio -- */

export function EquityCard({ portfolio, history }: {
  portfolio: Snapshot["portfolio"]; history: Snapshot["equity_history"];
}) {
  const tone = pnlTone(portfolio.daily_pnl);
  const spark = history.slice(-40).map((p) => p.equity);

  return (
    <Card span="1">
      <CardHeader>
        <CardTitle>Equity</CardTitle>
        {portfolio.daily_pnl !== 0 &&
          (portfolio.daily_pnl > 0
            ? <ArrowUpRight className="h-4 w-4 text-positive" aria-hidden />
            : <ArrowDownRight className="h-4 w-4 text-negative" aria-hidden />)}
      </CardHeader>
      <CardContent className="space-y-3">
        <Stat
          label="Account value" size="lg" value={money(portfolio.equity)}
          sub={`peak ${money(portfolio.peak_equity)}`}
        />
        <Sparkline data={spark} tone={tone === "muted" ? "primary" : tone} />
        <div className="grid grid-cols-2 gap-3">
          <Stat
            label="Today" size="sm" tone={tone}
            value={signedMoney(portfolio.daily_pnl)} sub={signedPct(portfolio.daily_pnl_pct)}
          />
          <Stat
            label="Buying power" size="sm"
            value={money(portfolio.buying_power)}
            sub={`${portfolio.daily_trades} trades today`}
          />
        </div>
      </CardContent>
    </Card>
  );
}

export function AllocationCard({ portfolio, risk }: {
  portfolio: Snapshot["portfolio"]; risk: Snapshot["risk"];
}) {
  const target = portfolio.target_allocation;
  const cap = risk.limits.max_exposure;
  const over = target !== null && target > cap;

  return (
    <Card span="1">
      <CardHeader>
        <CardTitle>Allocation</CardTitle>
        <Gauge className="h-4 w-4 text-muted-foreground" aria-hidden />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Invested" size="md" value={pct(portfolio.allocation, 0)} />
          <Stat
            label="Leverage" size="md" value={`${portfolio.leverage.toFixed(2)}x`}
            sub={`cap ${risk.limits.max_leverage.toFixed(2)}x`}
          />
        </div>
        <AllocationBar current={portfolio.allocation} target={target} cap={cap} />
      </CardContent>
      {over && (
        <CardFooter className="text-warning">
          The strategy wants {pct(target!, 0)} but the risk layer caps gross exposure at{" "}
          {pct(cap, 0)}. The veto wins, so the target is unreachable by construction.
        </CardFooter>
      )}
    </Card>
  );
}

/* -------------------------------------------------------------- risk -- */

function Meter({ name, used, limit, soft }: {
  name: string; used: number; limit: number; soft?: number;
}) {
  const tone = riskTone(used, limit);
  const ratio = limit ? Math.min(1, Math.abs(used) / Math.abs(limit)) : 0;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-muted-foreground">{name}</span>
        <span className={cn("tnum font-medium", toneText[tone])}>
          {pct(Math.abs(used), 2)}
          <span className="text-muted-foreground"> / {pct(Math.abs(limit), 0)}</span>
        </span>
      </div>
      <Progress value={ratio} tone={tone} mark={soft ? soft / limit : undefined} label={name} />
    </div>
  );
}

export function RiskCard({ risk }: { risk: Snapshot["risk"] }) {
  const dd = risk.drawdowns;
  return (
    <Card span="1">
      <CardHeader>
        <CardTitle>Risk status</CardTitle>
        {risk.halted
          ? <Badge variant="negative" dot pulse>Halted</Badge>
          : <Badge variant="positive" dot>Armed</Badge>}
      </CardHeader>
      <CardContent className="space-y-4">
        {dd ? (
          <>
            <Meter name="Daily" used={dd.daily} limit={risk.limits.daily_halt} soft={risk.limits.daily_reduce} />
            <Meter name="Weekly" used={dd.weekly} limit={risk.limits.weekly_halt} soft={risk.limits.weekly_reduce} />
            <Meter name="From peak" used={dd.from_peak} limit={risk.limits.max_from_peak} />
            <div className="grid grid-cols-2 gap-3 pt-1">
              <Stat label="Risk / trade" size="sm" value={pct(risk.limits.max_risk_per_trade, 2)} />
              <Stat label="Breakers fired" size="sm" value={risk.n_triggers} />
            </div>
          </>
        ) : (
          <Empty>No portfolio snapshot yet.</Empty>
        )}
      </CardContent>
      <CardFooter>
        Breakers fire on realised P&amp;L, never on what the model believes. Ticks mark where
        sizing halves; the bar ends where trading stops.
      </CardFooter>
    </Card>
  );
}

/* ------------------------------------------------------------ charts -- */

export function EquityChartCard({ snapshot }: { snapshot: Snapshot }) {
  const { equity_history: history, regime_history: regimes, portfolio } = snapshot;
  const first = history[0]?.equity ?? portfolio.equity;
  const change = first ? portfolio.equity / first - 1 : 0;

  return (
    <Card span="3">
      <CardHeader>
        <CardTitle>Equity and regime history</CardTitle>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full bg-primary" />Equity
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-0.5 w-3 rounded-full bg-muted-foreground" />Peak
          </span>
          <span className="tnum">{history.length} bars</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 flex flex-wrap gap-6">
          <Stat
            label="Since start" size="md" tone={pnlTone(change)}
            value={signedPct(change, 1)}
            sub={`from ${money(first)}`}
          />
          <Stat
            label="Max drawdown" size="md"
            tone={history.length ? "negative" : "muted"}
            value={history.length
              ? pct(Math.min(...history.map((p) => p.equity / p.peak - 1)), 1)
              : "—"}
          />
        </div>
        <EquityChart data={history} regimes={regimes} />
      </CardContent>
    </Card>
  );
}

export function RegimeMixCard({ mix }: { mix: Snapshot["regime_mix"] }) {
  return (
    <Card span="2">
      <CardHeader>
        <CardTitle>Regime mix</CardTitle>
        <Activity className="h-4 w-4 text-muted-foreground" aria-hidden />
      </CardHeader>
      <CardContent>
        <RegimeMix data={mix} />
      </CardContent>
      <CardFooter>
        Share of bars per regime. A model reporting seven states that sat in one of them
        for most of the window has not really found seven.
      </CardFooter>
    </Card>
  );
}

/* --------------------------------------------------------- positions -- */

export function PositionsCard({ positions }: { positions: Snapshot["positions"] }) {
  const naked = positions.filter((p) => !p.has_stop);
  return (
    <Card span="2">
      <CardHeader>
        <CardTitle>Positions</CardTitle>
        <Badge variant={positions.length ? "primary" : "default"}>
          {positions.length} open
        </Badge>
      </CardHeader>
      <CardContent>
        {naked.length > 0 && (
          <div className="mb-4 flex items-start gap-2 rounded-md border border-negative/30 bg-negative/10 p-3 text-sm text-negative">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>
              <strong>{naked.map((p) => p.symbol).join(", ")}</strong>{" "}
              {naked.length === 1 ? "has" : "have"} no protective stop. Place one or close it.
            </span>
          </div>
        )}

        {positions.length === 0 ? (
          <Empty>No open positions.</Empty>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Qty</TableHead>
                <TableHead>Entry</TableHead>
                <TableHead>Last</TableHead>
                <TableHead>Stop</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>P&amp;L</TableHead>
                <TableHead>Held</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((p) => (
                <TableRow key={p.symbol}>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">{p.symbol}</span>
                      {p.regime_changed && <Badge variant="warning">regime moved</Badge>}
                      {p.adopted && <Badge>adopted</Badge>}
                    </div>
                  </TableCell>
                  <TableCell className="tnum text-muted-foreground">{p.quantity}</TableCell>
                  <TableCell className="tnum text-muted-foreground">{price(p.entry_price)}</TableCell>
                  <TableCell className="tnum">{price(p.current_price)}</TableCell>
                  <TableCell className="tnum">
                    {p.has_stop && p.stop_loss !== null ? (
                      <span>
                        {price(p.stop_loss)}
                        <span className="text-muted-foreground"> ({pct(p.distance_to_stop_pct, 1)})</span>
                      </span>
                    ) : (
                      <span className="text-negative">none</span>
                    )}
                  </TableCell>
                  <TableCell className="tnum text-muted-foreground">{money(p.market_value)}</TableCell>
                  <TableCell className={cn("tnum font-medium", toneText[pnlTone(p.unrealised_pnl)])}>
                    {signedMoney(p.unrealised_pnl)}
                    <span className="opacity-70"> {signedPct(p.unrealised_pnl_pct, 1)}</span>
                  </TableCell>
                  <TableCell className="tnum text-muted-foreground">{p.held_for}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

/* ----------------------------------------------------------- signals -- */

export function SignalsCard({ signals }: { signals: Snapshot["signals"] }) {
  const rows = [...signals].reverse().slice(0, 10);
  return (
    <Card span="2">
      <CardHeader>
        <CardTitle>Recent signals</CardTitle>
        <Signal className="h-4 w-4 text-muted-foreground" aria-hidden />
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <Empty>No signals yet.</Empty>
        ) : (
          <ul className="space-y-0.5">
            {rows.map((s, i) => {
              const rejected = s.event === "signal_rejected";
              return (
                <li
                  key={`${s.timestamp}-${s.symbol}-${i}`}
                  className="flex items-center gap-3 rounded-md px-2 py-2 text-sm transition-colors hover:bg-muted/40"
                >
                  {rejected
                    ? <Ban className="h-3.5 w-3.5 shrink-0 text-negative" aria-hidden />
                    : <CircleDot className="h-3.5 w-3.5 shrink-0 text-positive" aria-hidden />}
                  <span className="tnum w-11 shrink-0 text-xs text-muted-foreground">
                    {clockTime(s.timestamp)}
                  </span>
                  <span className="w-14 shrink-0 font-semibold">{s.symbol ?? "—"}</span>
                  <span className="min-w-0 flex-1 truncate text-muted-foreground" title={s.message}>
                    {rejected
                      ? humanise(s.rejection_reason ?? "rejected")
                      : `${s.shares ?? 0} shares · ${money(s.notional ?? 0)}`}
                  </span>
                  <Badge variant={rejected ? "negative" : "positive"}>
                    {rejected ? "Rejected" : "Approved"}
                  </Badge>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
      <CardFooter>
        Rejections appear as prominently as approvals. A feed of only the trades that
        happened cannot tell you the system stopped trading three weeks ago.
      </CardFooter>
    </Card>
  );
}

/* ------------------------------------------------------------ system -- */

export function SystemCard({ system, risk }: {
  system: Snapshot["system"]; risk: Snapshot["risk"];
}) {
  const rows: { label: string; value: string; tone: Tone }[] = [
    {
      label: "Data feed",
      value: system.data_feed_healthy ? "Healthy" : "Down",
      tone: system.data_feed_healthy ? "positive" : "negative",
    },
    {
      label: "Broker API",
      value: system.broker_connected
        ? `${Math.round(system.api_latency_ms ?? 0)}ms`
        : "Disconnected",
      tone: system.broker_connected ? "positive" : "negative",
    },
    {
      label: "Model age",
      value: system.model_age_days === null ? "None" : `${system.model_age_days.toFixed(1)}d`,
      tone: (system.model_age_days ?? 0) > 7 ? "warning" : "positive",
    },
    {
      label: "Market",
      value: system.market_open ? "Open" : "Closed",
      tone: system.market_open ? "positive" : "muted",
    },
    { label: "Bars processed", value: String(system.bars_processed), tone: "muted" },
    {
      label: "Breaker",
      value: humanise(risk.breaker_now ?? "none"),
      tone: risk.breaker_now && risk.breaker_now !== "none" ? "warning" : "positive",
    },
  ];

  return (
    <Card span="2">
      <CardHeader>
        <CardTitle>System</CardTitle>
        {system.data_feed_healthy
          ? <Wifi className="h-4 w-4 text-positive" aria-hidden />
          : <WifiOff className="h-4 w-4 text-negative" aria-hidden />}
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        {rows.map((r) => (
          <Stat key={r.label} label={r.label} size="sm" tone={r.tone} value={r.value} />
        ))}
      </CardContent>
      <CardFooter className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <ShieldCheck className="h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>{system.symbols.length} symbols · {system.timeframe ?? "—"}</span>
      </CardFooter>
    </Card>
  );
}

/* ------------------------------------------------------------ verdict -- */

export function VerdictCard() {
  return (
    <Card span="full" className="border-warning/25 bg-warning/[0.04]">
      <CardContent className="flex items-start gap-3 p-5">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
        <p className="text-sm leading-relaxed text-muted-foreground">
          <strong className="font-semibold text-foreground">
            This strategy has no demonstrated edge.
          </strong>{" "}
          Out of sample it loses to buy-and-hold and to random allocation under identical
          risk rules, and its drawdown trips the peak circuit breaker about 5% of the way
          into the backtest. Every item on the project&apos;s validation checklist is
          unticked, including &ldquo;30 to 50 closed trades&rdquo; against zero. The
          dashboard is honest about what the system does; that is not the same as the
          system working. Paper only.
        </p>
      </CardContent>
    </Card>
  );
}
