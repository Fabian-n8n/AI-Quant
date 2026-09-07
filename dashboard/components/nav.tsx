"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, LayoutDashboard } from "lucide-react";
import { cn } from "@/lib/utils";

const ROUTES = [
  { href: "/", label: "Overview", icon: LayoutDashboard,
    hint: "Regime, allocation, what to buy" },
  { href: "/activity", label: "Activity", icon: Activity,
    hint: "Orders, positions, run history" },
] as const;

/** Sidebar on desktop, a row of tabs on mobile.
 *
 *  Two routes does not justify a collapsible drawer with a hamburger. The
 *  whole navigation fits on screen at every width, so it stays visible: a menu
 *  you have to open to see where you are is worse than no menu at two items. */
export function Nav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Sections"
      className="sticky top-0 z-20 border-b border-border bg-background/80 backdrop-blur
                 lg:fixed lg:inset-y-0 lg:left-0 lg:w-56 lg:border-b-0 lg:border-r"
    >
      <div className="flex items-center gap-1 px-3 py-2 lg:h-full lg:flex-col lg:items-stretch lg:gap-1 lg:px-3 lg:py-5">
        <div className="mr-3 hidden items-center gap-2 px-2 pb-5 lg:flex">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary/15 text-primary">
            <svg viewBox="0 0 32 32" className="h-4 w-4" aria-hidden>
              <path d="M7 21l5-6 4 3 9-10" stroke="currentColor" strokeWidth="3"
                    fill="none" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="text-sm font-semibold tracking-tight">regime-trader</span>
        </div>

        {ROUTES.map(({ href, label, icon: Icon, hint }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              title={hint}
              className={cn(
                // 44px min touch target, and a focus ring that is never removed.
                "group flex min-h-11 items-center gap-2.5 rounded-lg px-3 py-2 text-sm",
                "transition-colors focus-visible:outline-none focus-visible:ring-2",
                "focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                active
                  ? "bg-primary/15 font-medium text-primary"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden />
              <span>{label}</span>
            </Link>
          );
        })}

        <p className="mt-auto hidden px-3 text-[11px] leading-relaxed text-muted-foreground lg:block">
          Read-only. Nothing here can place or cancel an order.
        </p>
      </div>
    </nav>
  );
}
