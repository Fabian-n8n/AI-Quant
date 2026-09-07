"use client";

import * as React from "react";
import { AlertTriangle, CheckCircle2, CircleSlash, Clock, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { money, price, relativeTime, signedMoney, signedPct } from "@/lib/format";
import { isDemo, useSnapshot } from "@/lib/useSnapshot";
import type {
  ClosedPositionRow, OpenPositionRow, OrderRow, RunRow, Snapshot,
} from "@/lib/types";

const EMPTY = {
  orders: [], open_positions: [], closed_positions: [], runs: [],
  expectancy: { trades: 0, expectancy: 0, win_rate: 0, avg_win: 0, avg_loss: 0 },
};

/** The paper trading log: what was ordered, what is held, what closed.
 *
 *  ORDERS ARE NOT TRADES
 *  ---------------------
 *  The most important thing this page does is keep them apart. A cancelled
 *  order with filled_qty 0 moved no money and changed no position, and a table
 *  that lists it beside a fill implies otherwise. Thirty resting test orders
 *  once made a working system look broken for exactly this reason, so unfilled
 *  rows are dimmed and only fills show a fill price.
 *
 *  Read-only, deliberately. No button here cancels an order: a display that
 *  can act is no longer a display, and the deployed copy holds no credentials
 *  to act with. */
export default function ActivityPage() {
  const { snap, error } = useSnapshot();
  const [status, setStatus] = React.useState("all");
  const [days, setDays] = React.useState(30);

  if (error && !snap) return <Screen message={error} />;
  if (!snap) return <Screen message="Loading..." />;

  const activity = snap.activity ?? EMPTY;
  const cutoff = Date.now() - days * 86_400_000;

  const orders = activity.orders.filter((o) => {
    if (status !== "all" && o.status !== status) return false;
    const at = new Date(o.submitted_at).getTime();
    return Number.isNaN(at) || at >= cutoff;
  });

  const statuses = Array.from(new Set(activity.orders.map((o) => o.status))).sort();
  const anything =
    activity.orders.length + activity.open_positions.length +
    activity.closed_positions.length + activity.runs.length > 0;

  return (
    <main className="container py-6 sm:py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">Activity</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Every order, position and scheduled run, newest first. Read-only.
        </p>
      </header>

      {isDemo(snap) && (
        <div className="mb-6 rounded-lg border border-warning/30 bg-warning/10 p-4 text-sm">
          <strong className="font-semibold text-foreground">Demo data.</strong>{" "}
          <span className="text-muted-foreground">
            No orders have been placed. Run{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">
              python main.py --once --publish
            </code>{" "}
            to replace this with real history.
          </span>
        </div>
      )}

      {!anything && !isDemo(snap) && (
        <div className="mb-6 rounded-lg border border-border bg-card/80 p-5 text-sm text-muted-foreground">
          Nothing recorded yet. The database exists and is empty, which is what a
          system that has not traded looks like.
        </div>
      )}

      <Expectancy snap={snap} />

      <section className="mt-6 space-y-5">
        <OpenPositions rows={activity.open_positions} />
        <Orders
          rows={orders} total={activity.orders.length} statuses={statuses}
          status={status} onStatus={setStatus} days={days} onDays={setDays}
        />
        <ClosedPositions rows={activity.closed_positions} />
        <Runs rows={activity.runs} />
      </section>
    </main>
  );
}

/* ------------------------------------------------------------------ */

function Expectancy({ snap }: { snap: Snapshot }) {
  const e = snap.activity?.expectancy;
  if (!e) return null;

  // Expectancy alone hides its own shape: one outsized winner and nineteen
  // losers is a positive number and not a strategy. The win rate and both
  // average sizes sit beside it so the figure can be read honestly.
  const cells: { label: string; value: string; tone?: string; hint?: string }[] = [
    { label: "Closed trades", value: String(e.trades),
      hint: "30 needed before live is considered" },
    { label: "Expectancy / trade", value: e.trades ? signedMoney(e.expectancy, 2) : "--",
      tone: !e.trades ? "" : e.expectancy > 0 ? "text-positive" : "text-negative" },
    { label: "Win rate", value: e.trades ? `${(e.win_rate * 100).toFixed(0)}%` : "--" },
    { label: "Avg win", value: e.trades ? signedMoney(e.avg_win, 2) : "--" },
    { label: "Avg loss", value: e.trades ? signedMoney(e.avg_loss, 2) : "--" },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {cells.map((c) => (
        <div key={c.label}
             className="rounded-lg border border-border/70 bg-card/80 px-4 py-3 backdrop-blur-sm">
          <div className="text-2xs font-semibold uppercase tracking-[0.09em] text-muted-foreground">
            {c.label}
          </div>
          <div className={`mt-1.5 font-mono text-lg tabular-nums ${c.tone ?? ""}`}>
            {c.value}
          </div>
          {c.hint && (
            <div className="mt-0.5 text-2xs text-muted-foreground">{c.hint}</div>
          )}
        </div>
      ))}
    </div>
  );
}

function Panel({
  title, hint, right, children,
}: {
  title: string; hint?: string; right?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-baseline gap-2">
          <CardTitle>{title}</CardTitle>
          {hint && <span className="text-2xs text-muted-foreground">{hint}</span>}
        </div>
        {right}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function OpenPositions({ rows }: { rows: OpenPositionRow[] }) {
  return (
    <Panel title="Open positions" hint={`${rows.length} held`}>
      {rows.length === 0 ? (
        <Blank>No open positions.</Blank>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead><TableHead>Qty</TableHead>
              <TableHead>Entry</TableHead><TableHead>Current</TableHead>
              <TableHead>Unrealised</TableHead><TableHead>Stop</TableHead>
              <TableHead>To stop</TableHead><TableHead>Held</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((p) => {
              const last = p.current_price ?? p.entry_price;
              // Distance to stop says how much room is left, which a bare
              // "stop: 163.04" does not.
              const toStop = p.stop_price && last ? (last - p.stop_price) / last : null;
              const pnl = p.unrealised_pnl ?? 0;
              return (
                <TableRow key={`${p.symbol}-${p.entry_at}`}>
                  <TableCell className="font-medium">{p.symbol}</TableCell>
                  <TableCell className="font-mono tabular-nums">{p.quantity}</TableCell>
                  <TableCell className="font-mono tabular-nums">{price(p.entry_price)}</TableCell>
                  <TableCell className="font-mono tabular-nums">{price(last)}</TableCell>
                  <TableCell
                    className={`font-mono tabular-nums ${pnl >= 0 ? "text-positive" : "text-negative"}`}
                  >
                    {signedMoney(pnl, 2)}
                  </TableCell>
                  <TableCell className="font-mono tabular-nums">
                    {p.stop_price ? price(p.stop_price)
                      : <span className="text-negative">none</span>}
                  </TableCell>
                  <TableCell className="font-mono tabular-nums">
                    {toStop === null ? "--" : signedPct(-toStop)}
                  </TableCell>
                  <TableCell className="font-mono tabular-nums">
                    {p.holding_days ?? 0}d
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}

function Orders({
  rows, total, statuses, status, onStatus, days, onDays,
}: {
  rows: OrderRow[]; total: number; statuses: string[];
  status: string; onStatus: (v: string) => void;
  days: number; onDays: (v: number) => void;
}) {
  return (
    <Panel
      title="Orders"
      hint={`${rows.length} of ${total}`}
      right={
        <div className="flex flex-wrap items-center gap-2">
          <Select label="Status" value={status} onChange={onStatus}
                  options={[["all", "All"],
                            ...statuses.map((s) => [s, s] as [string, string])]} />
          <Select label="Range" value={String(days)} onChange={(v) => onDays(Number(v))}
                  options={[["7", "7 days"], ["30", "30 days"],
                            ["90", "90 days"], ["36500", "All time"]]} />
        </div>
      }
    >
      <p className="mb-3 text-xs text-muted-foreground">
        An order is an instruction, not a trade. Dimmed rows never filled: they
        moved no money and changed no position.
      </p>
      {rows.length === 0 ? (
        <Blank>No orders match this filter.</Blank>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>When</TableHead><TableHead>Symbol</TableHead>
              <TableHead>Side</TableHead><TableHead>Type</TableHead>
              <TableHead>Qty</TableHead><TableHead>Submitted</TableHead>
              <TableHead>Filled at</TableHead><TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((o, i) => {
              const filled = o.filled_qty > 0 && o.fill_price != null;
              return (
                <TableRow key={`${o.symbol}-${o.submitted_at}-${i}`}
                          className={filled ? undefined : "opacity-55"}>
                  <TableCell className="text-muted-foreground">
                    {relativeTime(o.submitted_at)}
                  </TableCell>
                  <TableCell className="font-medium">{o.symbol}</TableCell>
                  <TableCell className={o.side === "buy" ? "text-positive" : "text-negative"}>
                    {o.side}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{o.order_type}</TableCell>
                  <TableCell className="font-mono tabular-nums">{o.quantity}</TableCell>
                  <TableCell className="font-mono tabular-nums">
                    {o.submitted_price ? price(o.submitted_price) : "market"}
                  </TableCell>
                  <TableCell className="font-mono tabular-nums">
                    {filled ? price(o.fill_price as number)
                      : <span className="text-muted-foreground">not filled</span>}
                  </TableCell>
                  <TableCell>
                    <OrderStatus status={o.status} skipped={o.skipped_reason} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}

function ClosedPositions({ rows }: { rows: ClosedPositionRow[] }) {
  return (
    <Panel title="Closed positions" hint={`${rows.length} completed`}>
      {rows.length === 0 ? (
        <Blank>
          Nothing has closed yet. This is the table the 30-trade validation gate
          counts, and it is empty.
        </Blank>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Closed</TableHead><TableHead>Symbol</TableHead>
              <TableHead>Qty</TableHead><TableHead>Entry</TableHead>
              <TableHead>Exit</TableHead><TableHead>Realised</TableHead>
              <TableHead>Held</TableHead><TableHead>Why</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((p, i) => (
              <TableRow key={`${p.symbol}-${p.exit_at}-${i}`}>
                <TableCell className="text-muted-foreground">
                  {relativeTime(p.exit_at)}
                </TableCell>
                <TableCell className="font-medium">{p.symbol}</TableCell>
                <TableCell className="font-mono tabular-nums">{p.quantity}</TableCell>
                <TableCell className="font-mono tabular-nums">{price(p.entry_price)}</TableCell>
                <TableCell className="font-mono tabular-nums">{price(p.exit_price)}</TableCell>
                <TableCell
                  className={`font-mono tabular-nums ${
                    p.realised_pnl >= 0 ? "text-positive" : "text-negative"}`}
                >
                  {signedMoney(p.realised_pnl, 2)}
                </TableCell>
                <TableCell className="font-mono tabular-nums">{p.holding_days ?? 0}d</TableCell>
                <TableCell><ExitReason reason={p.exit_reason} /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}

function Runs({ rows }: { rows: RunRow[] }) {
  return (
    <Panel title="Run history" hint="scheduled and manual">
      {rows.length === 0 ? (
        <Blank>No runs recorded.</Blank>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Started</TableHead><TableHead>Mode</TableHead>
              <TableHead>Trigger</TableHead><TableHead>Bars</TableHead>
              <TableHead>Orders</TableHead><TableHead>Equity</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.id}>
                <TableCell className="text-muted-foreground">
                  {relativeTime(r.started_at)}
                </TableCell>
                <TableCell>{r.mode}</TableCell>
                <TableCell className="text-muted-foreground">{r.trigger ?? "--"}</TableCell>
                <TableCell className="font-mono tabular-nums">{r.bars_processed}</TableCell>
                <TableCell className="font-mono tabular-nums">{r.orders_submitted}</TableCell>
                <TableCell className="font-mono tabular-nums">
                  {r.equity ? money(r.equity) : "--"}
                </TableCell>
                <TableCell><RunStatus status={r.status} error={r.error} /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Panel>
  );
}

/* -- small pieces --------------------------------------------------- */

const STATUS_STYLE: Record<string, { tone: string; Icon: typeof CheckCircle2 }> = {
  filled: { tone: "text-positive", Icon: CheckCircle2 },
  partially_filled: { tone: "text-warning", Icon: Clock },
  canceled: { tone: "text-muted-foreground", Icon: XCircle },
  cancelled: { tone: "text-muted-foreground", Icon: XCircle },
  expired: { tone: "text-muted-foreground", Icon: XCircle },
  rejected: { tone: "text-negative", Icon: AlertTriangle },
};

function OrderStatus({ status, skipped }: { status: string; skipped: string | null }) {
  if (skipped) {
    return (
      <span title={skipped}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <CircleSlash className="h-3 w-3" aria-hidden /> skipped
      </span>
    );
  }
  const { tone, Icon } = STATUS_STYLE[status] ?? {
    tone: "text-muted-foreground", Icon: Clock,
  };
  return (
    <span className={`inline-flex items-center gap-1 text-xs ${tone}`}>
      <Icon className="h-3 w-3" aria-hidden /> {status}
    </span>
  );
}

/** Why a position ended. Stop versus target is the distinction that matters: a
 *  book full of stops and no targets is a strategy that is not working, and
 *  that stays invisible if every exit reads the same. */
function ExitReason({ reason }: { reason: string }) {
  const variant =
    reason === "target" ? "positive"
      : reason === "stop" || reason === "trailing_stop" ? "negative"
      : reason === "breaker" ? "warning"
      : "default";
  return <Badge variant={variant}>{reason.replace(/_/g, " ")}</Badge>;
}

function RunStatus({ status, error }: { status: string; error: string | null }) {
  const variant =
    status === "ok" ? "positive"
      : status === "failed" ? "negative"
      : status === "halted" ? "warning"
      : "default";
  // 'running' on a past run means the process died before it could write an
  // ending. That is not success, and it should not read like one.
  return (
    <span title={error ?? undefined}>
      <Badge variant={variant}>{status === "running" ? "incomplete" : status}</Badge>
    </span>
  );
}

function Select({
  label, value, onChange, options,
}: {
  label: string; value: string; onChange: (v: string) => void;
  options: [string, string][];
}) {
  const id = React.useId();
  return (
    <span className="flex items-center gap-1.5">
      <label htmlFor={id}
             className="text-2xs font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="min-h-9 rounded-md border border-border bg-muted/40 px-2 py-1 text-xs
                   text-foreground focus-visible:outline-none focus-visible:ring-2
                   focus-visible:ring-ring"
      >
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </span>
  );
}

const Blank = ({ children }: { children: React.ReactNode }) => (
  <p className="py-6 text-center text-sm text-muted-foreground">{children}</p>
);

const Screen = ({ message }: { message: string }) => (
  <main className="container grid min-h-[60vh] place-items-center">
    <p className="text-sm text-muted-foreground">{message}</p>
  </main>
);
