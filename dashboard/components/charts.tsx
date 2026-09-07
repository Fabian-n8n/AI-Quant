"use client";

import * as React from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { cn } from "@/lib/utils";
import { humanise, money, pct, type Tone, toneHsl } from "@/lib/format";

/* -------------------------------------------------------------- tooltip -- */

/** One tooltip for every chart, so hover looks the same everywhere. Recharts'
 *  default is a white box, which on this theme reads as a rendering bug. */
function ChartTooltip({ active, payload, label, formatter }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-card/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {label && <div className="mb-1 font-medium text-muted-foreground">{label}</div>}
      {payload.map((entry: any) => (
        <div key={entry.dataKey} className="tnum flex items-center gap-2 text-foreground">
          <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="ml-auto font-semibold">
            {formatter ? formatter(entry.value) : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/* --------------------------------------------------------- equity curve -- */

interface EquityPoint { t: string; equity: number; peak: number }

const bandFill = (regime: string) =>
  regime.includes("bear") || regime === "crash"
    ? "hsl(var(--negative) / 0.07)"
    : regime.includes("bull") || regime === "euphoria"
    ? "hsl(var(--positive) / 0.07)"
    : "hsl(var(--primary) / 0.05)";

/** Equity against its own high-water mark, with regime bands behind it.
 *
 *  The peak line is the point of the chart. Equity alone says where you are;
 *  equity against peak says how far below your best you are, and that is the
 *  number the circuit breaker actually reads. */
export function EquityChart({
  data, regimes,
}: { data: EquityPoint[]; regimes: { t: string; regime: string }[] }) {
  const bands = React.useMemo(() => {
    const out: { from: string; to: string; regime: string }[] = [];
    regimes.forEach((r) => {
      const last = out[out.length - 1];
      if (last && last.regime === r.regime) last.to = r.t;
      else out.push({ from: r.t, to: r.t, regime: r.regime });
    });
    return out;
  }, [regimes]);

  if (data.length < 2) {
    return (
      <div className="flex h-[230px] items-center justify-center text-sm text-muted-foreground">
        Not enough history to chart yet.
      </div>
    );
  }

  const values = data.flatMap((d) => [d.equity, d.peak]);
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.12, hi * 0.004);

  return (
    <ResponsiveContainer width="100%" height={230}>
      <AreaChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 4 }}>
        <defs>
          <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.35} />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
          </linearGradient>
        </defs>

        {bands.map((b, i) => (
          <ReferenceArea key={i} x1={b.from} x2={b.to} fill={bandFill(b.regime)} strokeOpacity={0} />
        ))}

        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" vertical={false} />
        <XAxis
          dataKey="t" tickLine={false} axisLine={false} minTickGap={48}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          tickFormatter={(v: string) => v.slice(5)}
        />
        <YAxis
          domain={[lo - pad, hi + pad]} tickLine={false} axisLine={false} width={54}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip content={<ChartTooltip formatter={(v: number) => money(v, 2)} />} />

        <Area
          type="monotone" dataKey="equity" name="Equity" stroke="hsl(var(--primary))"
          strokeWidth={2} fill="url(#equityFill)" dot={false} isAnimationActive={false}
        />
        <Line
          type="monotone" dataKey="peak" name="High-water mark"
          stroke="hsl(var(--muted-foreground))" strokeWidth={1} strokeDasharray="4 4"
          dot={false} isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------ regime mix bars -- */

/** How the model spent its time. A system that reports seven regimes but sat in
 *  one of them for 90% of the window has not really found seven. */
export function RegimeMix({ data }: { data: { regime: string; bars: number; pct: number }[] }) {
  if (!data.length) {
    return <p className="py-6 text-center text-sm text-muted-foreground">No regime history yet.</p>;
  }
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, data.length * 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 0 }}>
        <XAxis type="number" hide domain={[0, Math.max(...data.map((d) => d.pct))]} />
        <YAxis
          type="category" dataKey="regime" width={86} tickLine={false} axisLine={false}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          tickFormatter={humanise}
        />
        <Tooltip
          cursor={{ fill: "hsl(var(--muted) / 0.4)" }}
          content={<ChartTooltip formatter={(v: number) => pct(v, 1)} />}
        />
        <Bar dataKey="pct" name="Share of bars" radius={[3, 3, 3, 3]} barSize={13} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.regime} fill={
              d.regime.includes("bear") || d.regime === "crash"
                ? "hsl(var(--negative) / 0.75)"
                : d.regime.includes("bull") || d.regime === "euphoria"
                ? "hsl(var(--positive) / 0.75)"
                : "hsl(var(--primary) / 0.75)"
            } />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------- sparkline -- */

export function Sparkline({ data, tone = "primary" }: { data: number[]; tone?: Tone }) {
  if (data.length < 2) return null;
  const lo = Math.min(...data), hi = Math.max(...data), range = hi - lo || 1;
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * 100},${28 - ((v - lo) / range) * 26}`)
    .join(" ");
  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="h-7 w-full" aria-hidden>
      <polyline
        points={points} fill="none" stroke={toneHsl[tone]} strokeWidth="1.5"
        vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round"
      />
    </svg>
  );
}

/* --------------------------------------------------------------- gauges -- */

/** Confidence as a ring.
 *
 *  The HMM's probability is the one figure that changes how much every other
 *  number here should be trusted, so it gets a shape readable without parsing
 *  digits. Banded on the config's own 0.55 floor and 0.7 comfort level. */
export function ConfidenceGauge({ value, size = 132 }: { value: number; size?: number }) {
  const stroke = 9;
  const r = (size - stroke) / 2 - 3;
  const c = 2 * Math.PI * r;
  const tone: Tone = value >= 0.7 ? "positive" : value >= 0.55 ? "warning" : "negative";

  return (
    <svg
      width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img"
      aria-label={`Regime confidence ${Math.round(value * 100)} percent`}
      className="shrink-0"
    >
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="hsl(var(--muted))" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke={toneHsl[tone]} strokeWidth={stroke}
        strokeLinecap="round" strokeDasharray={`${c * value} ${c}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: "stroke-dasharray 500ms cubic-bezier(.22,.61,.36,1)" }}
      />
      <text
        x="50%" y="47%" textAnchor="middle" className="tnum"
        fontSize={size * 0.23} fontWeight={650} fill="hsl(var(--foreground))"
      >
        {Math.round(value * 100)}%
      </text>
      <text
        x="50%" y="64%" textAnchor="middle" fontSize={size * 0.075}
        letterSpacing="1.2" fill="hsl(var(--muted-foreground))"
      >
        CONFIDENCE
      </text>
    </svg>
  );
}

/** Allocation against its target and the hard exposure cap.
 *
 *  Three numbers on one axis because they are only meaningful relative to each
 *  other: a 60% allocation is fine against an 80% cap and a problem against a
 *  50% one. Shows the Phase 3/5 conflict plainly when a target exceeds the cap. */
export function AllocationBar({
  current, target, cap,
}: { current: number; target: number | null; cap: number }) {
  const scale = Math.max(current, target ?? 0, cap) * 1.08 || 1;
  const over = target !== null && target > cap;

  return (
    <div className="space-y-2">
      <div className="relative h-9 w-full overflow-hidden rounded-md bg-muted">
        <div
          className="h-full bg-primary/70 transition-[width] duration-500"
          style={{ width: `${(current / scale) * 100}%` }}
        />
        {target !== null && (
          <span
            className={cn("absolute inset-y-0 w-0.5", over ? "bg-negative" : "bg-foreground/60")}
            style={{ left: `${(target / scale) * 100}%` }}
            title={`target ${pct(target, 0)}`}
          />
        )}
        <span
          className="absolute inset-y-0 w-0.5 bg-warning"
          style={{ left: `${(cap / scale) * 100}%` }}
          title={`cap ${pct(cap, 0)}`}
        />
      </div>
      <div className="tnum flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-primary/70" />held {pct(current, 0)}
        </span>
        {target !== null && (
          <span className="flex items-center gap-1.5">
            <span className={cn("h-2 w-0.5", over ? "bg-negative" : "bg-foreground/60")} />
            target {pct(target, 0)}
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-0.5 bg-warning" />cap {pct(cap, 0)}
        </span>
      </div>
    </div>
  );
}
