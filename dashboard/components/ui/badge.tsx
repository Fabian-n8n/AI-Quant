import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-2xs font-semibold uppercase tracking-wider whitespace-nowrap transition-colors",
  {
    variants: {
      variant: {
        default: "border-border bg-muted text-muted-foreground",
        primary: "border-primary/35 bg-primary/10 text-primary",
        positive: "border-positive/35 bg-positive/10 text-positive",
        negative: "border-negative/35 bg-negative/10 text-negative",
        warning: "border-warning/35 bg-warning/10 text-warning",
        outline: "border-border text-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {
  /** A small filled circle before the label. Pulses for live states. */
  dot?: boolean;
  pulse?: boolean;
}

function Badge({ className, variant, dot, pulse, children, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props}>
      {dot && (
        <span
          aria-hidden
          className={cn("h-1.5 w-1.5 rounded-full bg-current", pulse && "animate-pulse-dot")}
        />
      )}
      {children}
    </span>
  );
}

export { Badge };
