"use client";

import * as React from "react";
import {
  ArrowRight, Ban, ChevronDown, CircleDollarSign, Shield, TrendingDown, TrendingUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Stat } from "@/components/ui/stat";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { humanise, money, pct, price, signedPct, toneText, type Tone } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Candidate } from "@/lib/types";

const convictionTone = (v: number): Tone =>
  v >= 0.7 ? "positive" : v >= 0.5 ? "warning" : "muted";

/* ----------------------------------------------------------- top pick -- */

/** The dashboard's primary answer: the one name the system would act on now.
 *
 *  Deliberately absent when nothing is approved. A halted breaker, a regime the
 *  strategy sits out, or an exposure cap already reached all produce "nothing",
 *  and showing the least-bad blocked name instead would invent a recommendation
 *  the system never made. */
export function TopPickCard({ pick, equity }: { pick: Candidate | null; equity: number }) {
  if (!pick) {
    return (
      <Card span="2">
        <CardHeader>
          <CardTitle>Strongest signal</CardTitle>
          <Badge>No action</Badge>
        </CardHeader>
        <CardContent className="flex items-center gap-3 py-8">
          <Shield className="h-5 w-5 shrink-0 text-muted-foreground" aria-hidden />
          <p className="text-sm text-muted-foreground">
            Nothing is approved to buy right now. That is a real answer, not a gap:
            the risk layer blocks every candidate when a breaker has tripped, the
            exposure cap is reached, or the regime does not support new risk.
          </p>
        </CardContent>
      </Card>
    );
  }

  const riskTone: Tone = pick.risk_pct_of_equity > 0.008 ? "warning" : "positive";

  return (
    <Card span="2" className="border-primary/30 bg-gradient-to-br from-primary/[0.07] to-transparent">
      <CardHeader>
        <CardTitle>Strongest signal · what to buy</CardTitle>
        <Badge variant="primary" dot pulse>Rank 1 of {pick.rank > 0 ? "universe" : "—"}</Badge>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-semibold tracking-tight">{pick.symbol}</span>
              {pick.trend === "above" ? (
                <TrendingUp className="h-5 w-5 text-positive" aria-hidden />
              ) : (
                <TrendingDown className="h-5 w-5 text-negative" aria-hidden />
              )}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {pick.strategy.replace(/Strategy$/, "")} · {humanise(pick.volatility_rank)} volatility regime
            </p>
          </div>

          <div className="flex items-center gap-2 text-lg font-semibold">
            <span className="rounded-md bg-positive/15 px-2.5 py-1 text-positive">
              BUY {pick.shares.toLocaleString()} shares
            </span>
            <ArrowRight className="h-4 w-4 text-muted-foreground" aria-hidden />
            <span className="tnum">{money(pick.notional)}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Entry" size="sm" value={price(pick.entry_price)} />
          <Stat
            label="Stop" size="sm" tone="negative"
            value={pick.stop_loss !== null ? price(pick.stop_loss) : "—"}
            sub={`${pct(pick.stop_distance_pct, 2)} away`}
          />
          <Stat
            label="Risk if stopped" size="sm" tone={riskTone}
            value={money(pick.risk_dollars)}
            sub={`${pct(pick.risk_pct_of_equity, 2)} of equity`}
          />
          <Stat
            label="Position size" size="sm"
            value={equity ? pct(pick.notional / equity, 1) : "—"}
            sub="of account"
          />
        </div>

        <div>
          <div className="mb-1.5 flex items-baseline justify-between text-xs">
            <span className="text-muted-foreground">Conviction</span>
            <span className={cn("tnum font-semibold", toneText[convictionTone(pick.conviction)])}>
              {pct(pick.conviction, 0)}
            </span>
          </div>
          <Progress value={pick.conviction} tone={convictionTone(pick.conviction)} label="Conviction" />
        </div>

        {pick.modifications.length > 0 && (
          <ul className="space-y-1 text-xs text-muted-foreground">
            {pick.modifications.map((m) => (
              <li key={m} className="flex gap-2">
                <Shield className="mt-0.5 h-3 w-3 shrink-0 text-warning" aria-hidden />
                <span>{m}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>

      <CardFooter>
        The regime is <strong className="font-medium text-foreground">market-wide</strong>, so this
        ranks which name the risk layer would commit most to — it is not a per-stock forecast.
        Differentiation comes from trend, stop distance, correlation and sector limits.
      </CardFooter>
    </Card>
  );
}

/* --------------------------------------------------------- watchlist -- */

/** Every symbol, ranked, with the full decision chain behind each row.
 *
 *  Rows expand rather than opening a dialog: on a monitoring surface you want to
 *  compare two names side by side, and a modal makes that impossible. */
export function CandidatesCard({ candidates }: { candidates: Candidate[] }) {
  const [open, setOpen] = React.useState<string | null>(null);
  const approved = candidates.filter((c) => c.approved).length;

  if (!candidates.length) {
    return (
      <Card span="full">
        <CardHeader><CardTitle>Watchlist</CardTitle></CardHeader>
        <CardContent>
          <p className="py-8 text-center text-sm text-muted-foreground">
            No scan yet. The watchlist is computed once per bar.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card span="full">
      <CardHeader>
        <CardTitle>Watchlist · ranked by conviction</CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant="positive">{approved} tradable</Badge>
          <Badge>{candidates.length - approved} blocked</Badge>
        </div>
      </CardHeader>

      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>#</TableHead>
              <TableHead>Symbol</TableHead>
              <TableHead>Action</TableHead>
              <TableHead>Conviction</TableHead>
              <TableHead>Trend</TableHead>
              <TableHead>Entry</TableHead>
              <TableHead>Stop</TableHead>
              <TableHead>Shares</TableHead>
              <TableHead>Notional</TableHead>
              <TableHead>Risk</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {candidates.map((c) => {
              const isOpen = open === c.symbol;
              return (
                <React.Fragment key={c.symbol}>
                  <TableRow
                    className="cursor-pointer"
                    onClick={() => setOpen(isOpen ? null : c.symbol)}
                    tabIndex={0}
                    role="button"
                    aria-expanded={isOpen}
                    aria-label={`${c.symbol} details`}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setOpen(isOpen ? null : c.symbol);
                      }
                    }}
                  >
                    <TableCell className="tnum text-muted-foreground">{c.rank}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{c.symbol}</span>
                        {c.held && <Badge variant="primary">held</Badge>}
                      </div>
                    </TableCell>
                    <TableCell>
                      {c.approved ? (
                        <Badge variant={c.action === "hold" ? "primary" : "positive"}>
                          {c.action}
                        </Badge>
                      ) : (
                        <Badge variant="negative">blocked</Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-2">
                        <span className={cn("tnum text-xs", toneText[convictionTone(c.conviction)])}>
                          {pct(c.conviction, 0)}
                        </span>
                        <div className="w-14">
                          <Progress value={c.conviction} tone={convictionTone(c.conviction)} />
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className={c.trend === "above" ? "text-positive" : "text-negative"}>
                      {c.trend === "above" ? "↑" : "↓"} {signedPct(c.price_vs_ema50, 1)}
                    </TableCell>
                    <TableCell className="tnum">{price(c.entry_price)}</TableCell>
                    <TableCell className="tnum text-muted-foreground">
                      {c.stop_loss !== null ? price(c.stop_loss) : "—"}
                    </TableCell>
                    <TableCell className="tnum">{c.shares || "—"}</TableCell>
                    <TableCell className="tnum">{c.notional ? money(c.notional) : "—"}</TableCell>
                    <TableCell className="tnum text-muted-foreground">
                      {c.risk_dollars ? pct(c.risk_pct_of_equity, 2) : "—"}
                    </TableCell>
                    <TableCell>
                      <ChevronDown
                        className={cn(
                          "h-4 w-4 text-muted-foreground transition-transform duration-200",
                          isOpen && "rotate-180",
                        )}
                        aria-hidden
                      />
                    </TableCell>
                  </TableRow>

                  {isOpen && (
                    <TableRow className="hover:bg-transparent">
                      <TableCell colSpan={11} className="bg-muted/25 p-0 text-left">
                        <CandidateDetail candidate={c} />
                      </TableCell>
                    </TableRow>
                  )}
                </React.Fragment>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>

      <CardFooter>
        Click any row for the full signal-to-decision chain. Blocked names are kept and ranked:
        knowing a strong setup was refused for correlation is more useful than not seeing it.
      </CardFooter>
    </Card>
  );
}

/* ------------------------------------------------------------ detail -- */

function CandidateDetail({ candidate: c }: { candidate: Candidate }) {
  const steps: { label: string; value: React.ReactNode; tone?: Tone }[] = [
    { label: "Regime", value: `${humanise(c.regime)} at ${pct(c.regime_confidence, 0)} confidence` },
    { label: "Volatility tier", value: `${humanise(c.volatility_rank)} → ${c.strategy}` },
    {
      label: "Trend filter",
      value: `${c.trend === "above" ? "Above" : "Below"} the 50 EMA by ${signedPct(c.price_vs_ema50, 2)}`,
      tone: c.trend === "above" ? "positive" : "negative",
    },
    { label: "Volatility (ATR)", value: `${pct(c.atr_pct, 2)} of price` },
    {
      label: "Stop placement",
      value: c.stop_loss !== null
        ? `${price(c.stop_loss)} — ${pct(c.stop_distance_pct, 2)} away, ${c.stop_atr_mult.toFixed(1)}× ATR`
        : "none",
    },
    { label: "20-bar momentum", value: signedPct(c.return_20d, 1), tone: c.return_20d >= 0 ? "positive" : "negative" },
  ];

  return (
    <div className="grid gap-6 p-5 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <h4 className="mb-3 text-2xs font-semibold uppercase tracking-[0.09em] text-muted-foreground">
          How the system reached this
        </h4>
        <dl className="space-y-2">
          {steps.map((s) => (
            <div key={s.label} className="flex items-baseline justify-between gap-4 text-sm">
              <dt className="text-muted-foreground">{s.label}</dt>
              <dd className={cn("tnum text-right", s.tone ? toneText[s.tone] : "text-foreground")}>
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
        {c.reasoning && (
          <p className="mt-3 border-t border-border/60 pt-3 text-xs leading-relaxed text-muted-foreground">
            {c.reasoning}
          </p>
        )}
      </div>

      <div>
        <h4 className="mb-3 text-2xs font-semibold uppercase tracking-[0.09em] text-muted-foreground">
          Risk verdict
        </h4>
        {c.approved ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-positive">
              <CircleDollarSign className="h-4 w-4" aria-hidden />
              Approved · {c.shares.toLocaleString()} shares
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Deploys" size="sm" value={money(c.notional)} />
              <Stat label="Risks" size="sm" tone="warning" value={money(c.risk_dollars)}
                    sub={pct(c.risk_pct_of_equity, 2)} />
            </div>
            {c.modifications.length > 0 ? (
              <ul className="space-y-1.5 text-xs text-muted-foreground">
                {c.modifications.map((m) => (
                  <li key={m} className="flex gap-2">
                    <Shield className="mt-0.5 h-3 w-3 shrink-0 text-warning" aria-hidden />
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted-foreground">
                Approved unmodified: no cap or limit reduced this position.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-negative">
              <Ban className="h-4 w-4" aria-hidden />
              {humanise(c.rejection_reason ?? "blocked")}
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">{c.reason}</p>
          </div>
        )}
      </div>
    </div>
  );
}
