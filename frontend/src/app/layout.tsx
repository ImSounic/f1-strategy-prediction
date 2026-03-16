import type { Metadata } from 'next'
import './globals.css'
import { ThemeProvider } from '@/lib/ThemeProvider'

export const metadata: Metadata = {
  title: 'F1 Strategy Optimizer',
  description: 'Monte Carlo simulation for optimal Formula 1 pit stop strategies',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{
          __html: `
            (function() {
              try {
                var theme = localStorage.getItem('f1-theme');
                if (theme === 'light') {
                  document.documentElement.classList.remove('dark');
                } else if (!theme && window.matchMedia('(prefers-color-scheme: light)').matches) {
                  document.documentElement.classList.remove('dark');
                }
              } catch(e) {}
            })();
          `,
        }} />
      </head>
      <body className="min-h-screen bg-f1-darker transition-colors duration-300">
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
