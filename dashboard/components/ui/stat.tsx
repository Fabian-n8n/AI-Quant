import * as React from "react";
import { cn } from "@/lib/utils";
import { type Tone, toneText } from "@/lib/format";

/** One labelled figure. The unit of this dashboard, so it exists once. */
export function Stat({
  label, value, sub, tone = "muted", size = "md", className,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: Tone;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="text-2xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "tnum mt-1 font-semibold tracking-tight",
          size === "sm" && "text-lg",
          size === "md" && "text-2xl",
          size === "lg" && "text-3xl",
          tone !== "muted" ? toneText[tone] : "text-foreground",
        )}
      >
        {value}
      </div>
      {sub && <div className={cn("tnum mt-0.5 text-xs", toneText[tone === "muted" ? "muted" : tone])}>{sub}</div>}
    </div>
  );
}
