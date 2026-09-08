import type { Config } from "tailwindcss";

/** shadcn/ui token convention: every colour is an HSL triplet in a CSS variable,
 *  so the whole palette is themeable from one block in globals.css and Tailwind
 *  utilities compose against semantic names rather than raw hex. */
const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    // No 2xl cap. The 1440px default centred the content inside the space left
    // of the sidebar, so on a wide screen the sidebar sat flush against the
    // edge while the panels floated in the middle with gutters on both sides.
    // A data-dense dashboard should use the width it has; capping it wastes
    // exactly the space the extra columns were for.
    container: { center: true, padding: { DEFAULT: "1rem", sm: "1.5rem", xl: "2rem" } },
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
