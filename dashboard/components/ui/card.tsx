import * as React from "react";
import { cn } from "@/lib/utils";

/** shadcn Card, with one addition: `span` drives the bento placement so a cell
 *  declares its own size next to its content rather than in the page's grid. */
const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & { span?: "1" | "2" | "3" | "full" }
>(({ className, span, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "group relative flex flex-col rounded-lg border border-border/70 bg-card/80",
      "shadow-[0_1px_0_0_hsl(var(--foreground)/0.04)_inset] backdrop-blur-sm",
      "transition-colors duration-200 hover:border-border animate-fade-up",
      span === "1" && "cell-1",
      span === "2" && "cell-2",
      span === "3" && "cell-3",
      span === "full" && "cell-full",
      className,
    )}
    {...props}
  />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center justify-between gap-3 p-5 pb-3", className)} {...props} />
  ),
);
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("text-2xs font-semibold uppercase tracking-[0.09em] text-muted-foreground", className)}
      {...props}
    />
  ),
);
CardTitle.displayName = "CardTitle";

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex-1 px-5 pb-5", className)} {...props} />
  ),
);
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("mt-auto border-t border-border/60 px-5 py-3 text-xs leading-relaxed text-muted-foreground", className)}
      {...props}
    />
  ),
);
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardTitle, CardContent, CardFooter };
