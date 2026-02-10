import type { Metadata } from 'next'
import './globals.css'

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
    <html lang="en">
      <body className="min-h-screen bg-f1-darker">
        {children}
      </body>
    </html>
  )
}
