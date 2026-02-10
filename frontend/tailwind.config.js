/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        f1: {
          red: '#E10600',
          dark: '#15151E',
          darker: '#0D0D14',
          card: '#1A1A26',
          border: '#2A2A3A',
          muted: '#8A8A9A',
          light: '#E8E8F0',
        }
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui'],
        body: ['var(--font-body)', 'system-ui'],
        mono: ['var(--font-mono)', 'monospace'],
      },
    },
  },
  plugins: [],
}
