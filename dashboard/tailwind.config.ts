import type { Config } from "tailwindcss";

/** shadcn/ui token convention: every colour is an HSL triplet in a CSS variable,
 *  so the whole palette is themeable from one block in globals.css and Tailwind
 *  utilities compose against semantic names rather than raw hex. */
const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    // Every breakpoint is 100%, explicitly.
    //
    // Deleting the `screens` override does NOT make the container full width:
    // Tailwind falls back to the default breakpoint scale, so it stays capped
    // at 1536px. That is why the first attempt at this looked unchanged. The
    // cap has to be overridden at every breakpoint, not removed.
    //
    // Full width matters here because the sidebar is fixed to the left edge.
    // A centred, capped container inside the remaining space puts a gutter on
    // both sides of the panels while the sidebar has none, which reads as the
    // whole page being misaligned rather than as a max-width.
    container: {
      center: true,
      padding: { DEFAULT: "1rem", sm: "1.5rem", xl: "2rem" },
      screens: {
        sm: "100%", md: "100%", lg: "100%", xl: "100%", "2xl": "100%",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
          muted: "hsl(var(--primary-muted))",
        },
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        // Semantic, and deliberately never purple: accent means "system state",
        // these mean money. One hue for both is how a dashboard stops being
        // readable at a glance.
        positive: { DEFAULT: "hsl(var(--positive))", muted: "hsl(var(--positive-muted))" },
        negative: { DEFAULT: "hsl(var(--negative))", muted: "hsl(var(--negative-muted))" },
        warning: { DEFAULT: "hsl(var(--warning))", muted: "hsl(var(--warning-muted))" },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: { "2xs": ["0.6875rem", { lineHeight: "1rem" }] },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
        "pulse-dot": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
      },
      animation: {
        "fade-up": "fade-up 340ms cubic-bezier(.22,.61,.36,1) both",
        "pulse-dot": "pulse-dot 2.4s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
