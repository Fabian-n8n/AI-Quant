"use client";

import * as React from "react";
import { AlertTriangle, Loader2, TerminalSquare, TrendingUp } from "lucide-react";
import {
  AllocationCard, EquityCard, EquityChartCard, FreshnessCard, PositionsCard,
  RegimeCard, RegimeMixCard, RiskCard, SignalsCard, SystemCard, VerdictCard,
} from "@/components/panels";
import { CandidatesCard, TopPickCard } from "@/components/panels/candidates";
import { Badge } from "@/components/ui/badge";
import { relativeTime } from "@/lib/format";
import { isDemo, lastSuccessfulRun, useSnapshot } from "@/lib/useSnapshot";
import type { Snapshot } from "@/lib/types";

/** Read-only view of a published snapshot.
 *
 *  No credentials, no broker connection, no route that can place an order. The
 *  engine runs on your machine with your Alpaca keys and writes JSON; this
 *  renders it. A display that can act is no longer a display. */
export default function Page() {
  const { snap, error } = useSnapshot();

  if (error && !snap) return <ErrorState message={error} />;
  if (!snap) return <LoadingState />;

  const demo = isDemo(snap);
  // The one name the system would act on. Absent when nothing is approved,
  // which is a real answer rather than a gap.
  const topPick =
    (snap.candidates ?? []).find((c) => c.approved && c.action === "buy") ?? null;

  return (
    <main className="container py-6 sm:py-8">
      <Masthead snap={snap} demo={demo} />

      <MarketStatus snap={snap} />

      {demo && (
        <Notice tone="warning">
          <strong className="font-semibold text-foreground">Demo data, not a real account.</strong>{" "}
          Every figure below is generated so the interface can be judged. Publish your own with{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
            python main.py --dry-run --once --publish
          </code>
          .
        </Notice>
      )}

      {snap.risk.halted && (
        <Notice tone="negative">
          <strong className="font-semibold text-foreground">
            Trading halted by a circuit breaker.
          </strong>{" "}
          Every signal is rejected until <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
            trading_halted.lock
          </code>{" "}
          is deleted by hand. The friction is deliberate: someone should look at what broke
          before the system can lose more.
        </Notice>
      )}

      {/* Bento grid, 12 columns; each card declares its own span.
          Order is deliberate: the actionable answer first, the state that
          justifies it second, the machinery last. */}
      <div className="grid grid-cols-12 gap-4">
        <TopPickCard pick={topPick} equity={snap.portfolio.equity} timing={snap.timing} />
        <EquityCard portfolio={snap.portfolio} history={snap.equity_history} />
        <AllocationCard portfolio={snap.portfolio} risk={snap.risk} />

        <CandidatesCard candidates={snap.candidates ?? []} />

        <RegimeCard regime={snap.regime} />
        <RiskCard risk={snap.risk} />

        <EquityChartCard snapshot={snap} />
        <RegimeMixCard mix={snap.regime_mix} />

        <PositionsCard positions={snap.positions} />
        <SignalsCard signals={snap.signals} />

        <FreshnessCard freshness={snap.freshness} />
        <SystemCard system={snap.system} risk={snap.risk} />
        <VerdictCard />
      </div>

      <footer className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-5 text-xs text-muted-foreground">
        <span>Read-only. No credentials, no broker connection, no order path.</span>
        <span className="tnum">
          Snapshot {new Date(snap.timestamp).toISOString().slice(0, 19).replace("T", " ")} UTC
        </span>
      </footer>
    </main>
  );
}

/* ------------------------------------------------------------ chrome -- */

function Masthead({ snap, demo }: { snap: Snapshot; demo: boolean }) {
  return (
    <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <span
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-primary to-[hsl(263_70%_45%)] shadow-lg shadow-primary/25"
          aria-hidden
        >
          <TrendingUp className="h-5 w-5 text-primary-foreground" strokeWidth={2.5} />
        </span>
        <div>
          <h1 className="text-base font-semibold leading-tight tracking-tight">regime-trader</h1>
          <p className="text-xs text-muted-foreground">
            HMM regime detection · volatility-based allocation
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {demo
          ? <Badge variant="warning" dot>Demo data</Badge>
          : <Badge variant="primary" dot pulse>Live snapshot</Badge>}
        <Badge variant={snap.system.paper ? "positive" : "negative"}>
          {snap.system.paper ? "Paper" : "Live money"}
        </Badge>
        {snap.risk.halted && <Badge variant="negative" dot pulse>Halted</Badge>}
        <LastRun snap={snap} />
      </div>
    </header>
  );
}

/** When the system last completed a run, not when this file was written.
 *
 *  Those differ in the case that matters: a run that crashed still publishes,
 *  so "updated 2m ago" off the file timestamp would report a failure as
 *  freshness. Reading the last row with status 'ok' cannot do that. */
function LastRun({ snap }: { snap: Snapshot }) {
  const run = lastSuccessfulRun(snap);
  if (!run) {
    return <Badge variant="outline">Published {relativeTime(snap.published_at)}</Badge>;
  }
  const failedSince = (snap.activity?.runs ?? []).findIndex((r) => r.status === "ok") > 0;
  return (
    <Badge variant={failedSince ? "warning" : "outline"}
           title={failedSince ? "A more recent run did not complete" : undefined}>
      Last run {relativeTime(run.finished_at ?? run.started_at)}
    </Badge>
  );
}

/** Is the market open, and if not, when does it open.
 *
 *  An order resting unfilled looks identical to a broken system unless the
 *  page says the market has not opened yet. Nothing else on the dashboard
 *  answers "why has nothing happened", and it is the first question worth
 *  answering, because most of the time the answer is "it is 4am in New York". */
function MarketStatus({ snap }: { snap: Snapshot }) {
  const session = snap.timing?.session_state;
  if (!session) return null;

  const open = session.state === "open";
  const tone = open
    ? "border-positive/30 bg-positive/[0.07]"
    : session.state === "unknown"
      ? "border-warning/30 bg-warning/[0.07]"
      : "border-border bg-muted/20";

  return (
    <div className={`mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border
                     px-4 py-2.5 text-sm ${tone}`}>
      <span className="flex items-center gap-2 font-medium text-foreground">
        <span aria-hidden
              className={`h-2 w-2 rounded-full ${
                open ? "animate-pulse-dot bg-positive"
                     : session.state === "unknown" ? "bg-warning" : "bg-muted-foreground"}`} />
        {session.label}
      </span>
      <span className="text-muted-foreground">{session.detail}</span>
      {!open && (
        <span className="ml-auto text-xs text-muted-foreground">
          Refreshes pause outside market hours.
        </span>
      )}
    </div>
  );
}

function Notice({ tone, children }: { tone: "warning" | "negative"; children: React.ReactNode }) {
  return (
    <div
      className={
        "mb-4 flex items-start gap-3 rounded-lg border p-4 text-sm leading-relaxed text-muted-foreground " +
        (tone === "warning"
          ? "border-warning/30 bg-warning/[0.07]"
          : "border-negative/30 bg-negative/[0.07]")
      }
    >
      <AlertTriangle
        className={"mt-0.5 h-4 w-4 shrink-0 " + (tone === "warning" ? "text-warning" : "text-negative")}
        aria-hidden
      />
      <p>{children}</p>
    </div>
  );
}

function LoadingState() {
  return (
    <main className="container flex min-h-screen items-center justify-center">
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        Loading snapshot…
      </div>
    </main>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <main className="container flex min-h-screen items-center justify-center">
      <div className="max-w-lg rounded-lg border border-border bg-card p-6">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
          <TerminalSquare className="h-4 w-4 text-primary" aria-hidden />
          No snapshot found
        </div>
        <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
          The dashboard reads <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
            public/data/state.json
          </code>{" "}
          and nothing else. Generate one:
        </p>
        <pre className="overflow-x-auto rounded-md border border-border bg-background p-3 text-xs leading-relaxed text-muted-foreground">
{`python main.py --publish-demo                 # sample data
python main.py --dry-run --once --publish     # your own account`}
        </pre>
        <p className="mt-3 text-xs text-muted-foreground">({message})</p>
      </div>
    </main>
  );
}
