"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { type Tone, toneHsl } from "@/lib/format";

interface ProgressProps {
  value: number;             // 0..1
  tone?: Tone;
  /** Optional soft threshold, drawn as a tick. Where sizing halves, before the
   *  bar's end where trading stops. */
  mark?: number;
  className?: string;
  label?: string;
}

/** A meter, not a loading bar: it reports a level rather than progress toward
 *  completion, so it carries role="meter" and aria bounds. */
export function Progress({ value, tone = "primary", mark, className, label }: ProgressProps) {
  const ratio = Math.max(0, Math.min(1, value));
  return (
    <div
      role="meter"
      aria-valuenow={Math.round(ratio * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <div
        className="h-full rounded-full transition-[width] duration-500 ease-out"
        style={{ width: `${ratio * 100}%`, backgroundColor: toneHsl[tone] }}
      />
      {mark !== undefined && mark > 0 && mark < 1 && (
        <span
          aria-hidden
          className="absolute inset-y-0 w-px bg-foreground/45"
          style={{ left: `${mark * 100}%` }}
        />
      )}
    </div>
  );
}
