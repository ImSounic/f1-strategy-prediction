/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        f1: {
          red: '#E10600',
          dark: 'var(--f1-dark)',
          darker: 'var(--f1-darker)',
          card: 'var(--f1-card)',
          border: 'var(--f1-border)',
          muted: 'var(--f1-muted)',
          light: 'var(--f1-light)',
        }
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui'],
        body: ['var(--font-body)', 'system-ui'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      boxShadow: {
        'card': 'var(--card-shadow)',
        'card-hover': 'var(--card-shadow-hover)',
      },
    },
  },
  plugins: [],
}
