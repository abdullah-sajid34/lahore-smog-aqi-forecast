/** @type {import("tailwindcss").Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "#0B1220",
        card: "#111A2E",
        "card-light": "#F0F4FF",
        "bg-light": "#F7F9FC",
        accent: "#22D3EE",
        violet: "#8B5CF6",
        border: "rgba(34,211,238,0.12)",
        aqi: {
          good:            "#00E400",
          moderate:        "#FFD60A",
          sensitive:       "#FF7E00",
          unhealthy:       "#FF2D2D",
          veryUnhealthy:   "#8F3F97",
          hazardous:       "#7E0023",
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      borderRadius: { xl: "16px", "2xl": "24px" },
      backdropBlur: { xs: "2px" },
      keyframes: {
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        pulse2: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.5" } },
        countUp: { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
      animation: {
        shimmer: "shimmer 1.6s linear infinite",
        pulse2: "pulse2 2s ease-in-out infinite",
        countUp: "countUp 0.4s ease-out forwards",
      },
    },
  },
  plugins: [],
};