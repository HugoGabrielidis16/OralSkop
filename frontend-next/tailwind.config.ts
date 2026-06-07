import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        teal: "#0F6E56",
        deepTeal: "#0A4D3C",
        lightTeal: "#5EC9A8",
        coral: "#D85A30",
        coralTint: "#FDEDE6",
        cream: "#FAF9F6",
        charcoal: "#1F2937",
        gray: "#6B7280",
        line: "#E5E7EB",
      },
      fontFamily: {
        inter: ["Inter", "sans-serif"],
      },
      borderRadius: {
        card: "16px",
        pill: "100px",
      },
    },
  },
  plugins: [],
};

export default config;
