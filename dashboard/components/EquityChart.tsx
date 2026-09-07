"use client";

import { useId, useMemo, useState } from "react";
import { money } from "@/lib/format";

interface Point { t: string; equity: number; peak: number }

/** Equity against its own high-water mark, with regime bands underneath.
 *
 *  Hand-drawn SVG rather than a charting library. The whole chart is one path
 *  and a handful of rects; pulling in 40kB of recharts to draw that would be a
 *  larger dependency than the rest of the app combined, and this way the page
 *  ships no third-party JavaScript at all.
 *
 *  The peak line is the point of the chart. Equity alone tells you where you
 *  are; equity against peak tells you how far below your best you are, which is
 *  the number the circuit breaker actually reads. */
export default function EquityChart({
  data, regimes,
}: { data: Point[]; regimes: { t: string; regime: string }[] }) {
  const id = useId();
  const [hover, setHover] = useState<number | null>(null);

  const W = 760, H = 220, PAD_L = 8, PAD_R = 8, PAD_T = 12, PAD_B = 18;

  const view = useMemo(() => {
    if (data.length < 2) return null;
    const values = data.flatMap((d) => [d.equity, d.peak]);
    const lo = Math.min(...values), hi = Math.max(...values);
    // A 6% floor on the range stops a flat week from rendering as dramatic noise.
    const pad = Math.max((hi - lo) * 0.12, hi * 0.005);
    const min = lo - pad, max = hi + pad;

    const x = (i: number) => PAD_L + (i / (data.length - 1)) * (W - PAD_L - PAD_R);
    const y = (v: number) => PAD_T + (1 - (v - min) / (max - min || 1)) * (H - PAD_T - PAD_B);

    const line = data.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(d.equity).toFixed(2)}`).join("");
    const area = `${line}L${x(data.length - 1).toFixed(2)},${H - PAD_B}L${x(0).toFixed(2)},${H - PAD_B}Z`;
    const peak = data.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(d.peak).toFixed(2)}`).join("");

    // Contiguous runs of one regime, so the bands are blocks not stripes.
    const bands: { from: number; to: number; regime: string }[] = [];
    regimes.forEach((r, i) => {
      const last = bands[bands.length - 1];
      if (last && last.regime === r.regime) last.to = i;
      else bands.push({ from: i, to: i, regime: r.regime });
    });

    return { x, y, line, area, peak, min, max, bands };
  }, [data, regimes]);

  if (!view) {
    return <p className="faint" style={{ fontSize: 13 }}>Not enough history to chart yet.</p>;
  }

  const active = hover !== null ? data[hover] : data[data.length - 1];
  const first = data[0].equity;
  const change = (active.equity - first) / first;
  const fromPeak = active.equity / active.peak - 1;

  const bandFill = (regime: string) =>
    regime.includes("bear") || regime === "crash"
      ? "rgba(251,113,133,.075)"
      : regime.includes("bull") || regime === "euphoria"
      ? "rgba(52,211,153,.07)"
      : "rgba(167,139,250,.06)";

  return (
    <figure>
      <div style={{ display: "flex", gap: "var(--s6)", flexWrap: "wrap", marginBottom: "var(--s4)" }}>
        <div>
          <div className="stat-label">Equity</div>
          <div className="stat-value num">{money(active.equity)}</div>
        </div>
        <div>
          <div className="stat-label">Since start</div>
          <div className={`stat-value num ${change >= 0 ? "pos" : "neg"}`}>
            {change >= 0 ? "+" : ""}{(change * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="stat-label">From peak</div>
          <div className={`stat-value num ${fromPeak < -0.001 ? "neg" : "dim"}`}>
            {(fromPeak * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <svg
        className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        role="img"
        aria-label={`Equity curve, ${data.length} bars, currently ${money(active.equity)}, ${(fromPeak * 100).toFixed(1)} percent from peak`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const ratio = (e.clientX - box.left) / box.width;
          setHover(Math.max(0, Math.min(data.length - 1, Math.round(ratio * (data.length - 1)))));
        }}
      >
        <defs>
          <linearGradient id={`${id}-fill`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.34" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {view.bands.map((b, i) => (
          <rect
            key={i} x={view.x(b.from)} y={PAD_T}
            width={Math.max(1, view.x(b.to) - view.x(b.from))} height={H - PAD_T - PAD_B}
            fill={bandFill(b.regime)}
          />
        ))}

        <path d={view.area} fill={`url(#${id}-fill)`} />
        <path d={view.peak} fill="none" stroke="var(--fg-faint)" strokeWidth="1"
              strokeDasharray="3 4" opacity="0.6" vectorEffect="non-scaling-stroke" />
        <path d={view.line} fill="none" stroke="var(--accent)" strokeWidth="1.75"
              strokeLinejoin="round" vectorEffect="non-scaling-stroke" />

        {hover !== null && (
          <g>
            <line x1={view.x(hover)} y1={PAD_T} x2={view.x(hover)} y2={H - PAD_B}
                  stroke="var(--accent-line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            <circle cx={view.x(hover)} cy={view.y(data[hover].equity)} r="3"
                    fill="var(--accent)" stroke="var(--bg)" strokeWidth="1.5" />
          </g>
        )}
      </svg>

      <figcaption className="chart-legend">
        <span className="legend-key">
          <span className="legend-swatch" style={{ background: "var(--accent)" }} /> Equity
        </span>
        <span className="legend-key">
          <span className="legend-swatch" style={{ background: "var(--fg-faint)" }} /> High-water mark
        </span>
        <span className="legend-key">
          <span className="legend-swatch" style={{ background: "rgba(52,211,153,.5)" }} /> Bull regime
        </span>
        <span className="legend-key">
          <span className="legend-swatch" style={{ background: "rgba(251,113,133,.5)" }} /> Bear regime
        </span>
        <span style={{ marginLeft: "auto" }}>{active.t}</span>
      </figcaption>
    </figure>
  );
}
