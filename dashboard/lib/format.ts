/** Formatters. Every figure on this page goes through one of these.
 *
 *  Centralised because a dashboard that shows "$105230" in one panel and
 *  "$105,230.00" in another reads as two systems, and because the rules for
 *  what gets a sign and what gets decimals are decisions, not defaults. */

export const money = (v: number, decimals = 0) =>
  v.toLocaleString("en-US", {
    style: "currency", currency: "USD",
    minimumFractionDigits: decimals, maximumFractionDigits: decimals,
  });

/** Signed, for anything that is a change rather than a level. The explicit
 *  plus matters: "+340" and "340" carry different information at a glance. */
export const signedMoney = (v: number, decimals = 0) =>
  (v >= 0 ? "+" : "") + money(v, decimals);

export const pct = (v: number, decimals = 1) =>
  `${(v * 100).toFixed(decimals)}%`;

export const signedPct = (v: number, decimals = 2) =>
  `${v >= 0 ? "+" : ""}${(v * 100).toFixed(decimals)}%`;

export const price = (v: number) =>
  v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** "strong_bull" -> "Strong bull". Regime ids are snake_case in Python and
 *  should not leak that into the interface. */
export const humanise = (v: string | null | undefined) =>
  !v ? "Unknown" : v.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const clockTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? "--:--"
    : d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
};

export const relativeTime = (iso: string | null | undefined) => {
  if (!iso) return "never";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (Number.isNaN(seconds)) return "unknown";
  if (seconds < 90) return "just now";
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 172800) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
};

export const tone = (v: number) => (v > 0 ? "pos" : v < 0 ? "neg" : "dim");

/** Colour for a drawdown, graded on the FRACTION OF ITS LIMIT consumed, not on
 *  the drawdown itself. 2% of a 3% limit is nearly spent; 2% of a 10% limit has
 *  room. Grading the raw number would colour both the same, which is backwards. */
export const riskTone = (used: number, limit: number) => {
  if (!limit) return "dim";
  const ratio = Math.abs(used) / Math.abs(limit);
  return ratio < 0.5 ? "pos" : ratio < 0.8 ? "warn" : "neg";
};

export const toneColor = (t: string) =>
  ({ pos: "var(--pos)", neg: "var(--neg)", warn: "var(--warn)", dim: "var(--fg-dim)" }[t] ??
    "var(--fg-dim)");
