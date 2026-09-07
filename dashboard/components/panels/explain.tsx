"use client";

import * as React from "react";
import { Calculator, ChevronDown, Dices, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

/** Plain-language definitions for the four figures that decide a trade.
 *
 *  Written out because "Risk if stopped" is not self-evident, and a number
 *  nobody can define is a number nobody should act on. Collapsed by default so
 *  it does not compete with the figures themselves. */
const TERMS: { term: string; short: string; long: string }[] = [
  {
    term: "Entry",
    short: "the price the plan assumes you pay",
    long: "The most recent closing price. Your real fill will differ: the order goes in as a limit " +
      "a touch above this, and it executes at the next session's open, not now.",
  },
  {
    term: "Stop",
    short: "the price where you get out for a loss",
    long: "A resting sell order placed at the broker the moment the position fills. If the price " +
      "reaches it, the position is sold automatically. It is set from the 50-day EMA and the " +
      "stock's own recent volatility (ATR), then forced to sit strictly below entry. It only ever " +
      "moves up, never down — a stop that can retreat is not a stop.",
  },
  {
    term: "Risk if stopped",
    short: "what this trade costs you if the stop is hit",
    long: "Shares x (entry - stop). This is the number the position size is built around, not an " +
      "afterthought: the system picks how many shares to buy so that this figure stays under a " +
      "fixed fraction of the account. A wider stop therefore means fewer shares, not more risk.",
  },
  {
    term: "Conviction",
    short: "how well the setup fits the system's rules",
    long: "A 0-100% blend of four things: whether price is above its 50-day average, how tight the " +
      "stop is relative to the stock's volatility, how much of the requested size survived the " +
      "risk checks, and 20-day momentum. It is a ranking aid, not a probability, and it is not " +
      "calibrated against any outcome.",
  },
];

export function Glossary() {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="rounded-lg border border-border/70 bg-muted/20">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-medium transition-colors hover:bg-muted/30"
      >
        <HelpCircle className="h-4 w-4 shrink-0 text-primary" aria-hidden />
        What do Entry, Stop and Risk actually mean?
        <ChevronDown
          className={cn("ml-auto h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
          aria-hidden
        />
      </button>
      {open && (
        <dl className="space-y-4 border-t border-border/60 px-4 py-4">
          {TERMS.map((t) => (
            <div key={t.term}>
              <dt className="text-sm font-semibold">
                {t.term}
                <span className="ml-2 font-normal text-muted-foreground">— {t.short}</span>
              </dt>
              <dd className="mt-1 text-xs leading-relaxed text-muted-foreground">{t.long}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/** Splits an inputs list into what is arithmetic and what is a guess.
 *
 *  A trading dashboard that presents a model's probability estimate in the same
 *  visual language as a division sum invites you to trust both equally. The
 *  position size is arithmetic and will be exactly right; the regime is an
 *  estimate from a model fitted by an optimiser that finds local optima. */
export function DeterminismSplit({
  deterministic, probabilistic,
}: {
  deterministic: { label: string; value: React.ReactNode }[];
  probabilistic: { label: string; value: React.ReactNode }[];
}) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <section>
        <h5 className="mb-2 flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-[0.09em] text-positive">
          <Calculator className="h-3.5 w-3.5" aria-hidden />
          Deterministic
        </h5>
        <dl className="space-y-1.5">
          {deterministic.map((r) => (
            <div key={r.label} className="flex justify-between gap-3 text-xs">
              <dt className="text-muted-foreground">{r.label}</dt>
              <dd className="tnum text-right">{r.value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-2xs leading-relaxed text-muted-foreground">
          Arithmetic from config and price. Reproducible, and exactly right.
        </p>
      </section>

      <section>
        <h5 className="mb-2 flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-[0.09em] text-warning">
          <Dices className="h-3.5 w-3.5" aria-hidden />
          Estimated
        </h5>
        <dl className="space-y-1.5">
          {probabilistic.map((r) => (
            <div key={r.label} className="flex justify-between gap-3 text-xs">
              <dt className="text-muted-foreground">{r.label}</dt>
              <dd className="tnum text-right">{r.value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2 text-2xs leading-relaxed text-muted-foreground">
          Model output. The HMM is fitted by an optimiser that finds local optima,
          so a refit on the same data can label things differently.
        </p>
      </section>
    </div>
  );
}
