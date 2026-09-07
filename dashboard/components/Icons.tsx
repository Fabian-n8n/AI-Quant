/** Inline SVG, never emoji.
 *
 *  Emoji render differently on every platform, cannot inherit colour, and read
 *  as decoration in a surface where every mark should mean something. These are
 *  stroke icons on currentColor so they take the tone of whatever they sit in.
 *
 *  Each is `aria-hidden`: all of them sit beside a text label, so announcing
 *  them again would make a screen reader say everything twice. */

const base = {
  width: 14, height: 14, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 2,
  strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const Pulse = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 14} height={p.size ?? 14}>
    <path d="M3 12h4l3 8 4-16 3 8h4" />
  </svg>
);

export const Shield = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 14} height={p.size ?? 14}>
    <path d="M12 3l7 3v6c0 4.5-3 8-7 9-4-1-7-4.5-7-9V6z" />
  </svg>
);

export const Layers = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 14} height={p.size ?? 14}>
    <path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" />
  </svg>
);

export const Info = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 14} height={p.size ?? 14}>
    <circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" />
  </svg>
);

export const Warning = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 14} height={p.size ?? 14}>
    <path d="M12 4l9 16H3z" /><path d="M12 10v4M12 17h.01" />
  </svg>
);

export const Empty = (p: { size?: number }) => (
  <svg {...base} width={p.size ?? 26} height={p.size ?? 26} strokeWidth={1.4}>
    <rect x="3" y="5" width="18" height="14" rx="2" /><path d="M3 10h18" />
  </svg>
);

export const Mark = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff"
       strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 17l5-6 4 3 7-9" />
  </svg>
);
