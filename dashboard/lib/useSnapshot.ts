"use client";

import * as React from "react";
import type { Snapshot } from "@/lib/types";

/** Load the published snapshot, and keep it fresh.
 *
 *  Shared by every page so there is one fetch policy rather than one per
 *  route. Two pages polling the same file on different intervals would show
 *  different numbers to the same person at the same moment, which is the kind
 *  of inconsistency that makes a dashboard untrustworthy for no good reason.
 *
 *  The 5s cadence matches the terminal dashboard. It does NOT mean the data is
 *  live: the engine republishes once per processed bar, so between bars this
 *  polls an unchanged file. See the freshness panel, which exists to say so. */
export function useSnapshot() {
  const [snap, setSnap] = React.useState<Snapshot | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        // Absolute, not relative. A relative path resolves against the current
        // route, so on /activity/ it would ask for /activity/data/state.json
        // and 404. Cache-busted because the file is republished in place and a
        // cached read shows a stale account for as long as the browser likes.
        const res = await fetch(`/data/state.json?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as Snapshot;
        if (!cancelled) { setSnap(json); setError(null); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "could not load snapshot");
      }
    };

    load();
    const timer = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  return { snap, error };
}

/** Is this fabricated sample data?
 *
 *  Prefers the publisher's boolean and falls back to the source label, so an
 *  older published file still shows the banner rather than silently passing
 *  demo numbers off as an account. */
export const isDemo = (snap: Snapshot) =>
  snap.is_demo ?? snap.source === "demo";

/** When the system last completed a run, as opposed to when the file was
 *  written. A publish that happened during a failed run is not freshness. */
export const lastSuccessfulRun = (snap: Snapshot) =>
  (snap.activity?.runs ?? []).find((r) => r.status === "ok") ?? null;
