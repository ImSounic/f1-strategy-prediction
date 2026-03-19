'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useTheme } from '@/lib/ThemeProvider'

function ThemeToggle() {
  const { isDark, toggle } = useTheme()
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  if (!mounted) return <div className="w-9 h-9" />

  return (
    <button
      onClick={toggle}
      className="w-9 h-9 flex items-center justify-center rounded-lg border border-f1-border hover:border-f1-red text-f1-muted hover:text-f1-red transition-all"
      aria-label="Toggle theme"
    >
      {isDark ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  )
}

export function Header() {
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()
  const isSimulate = pathname === '/simulate'

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <header
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-f1-darker/95 backdrop-blur-xl border-b border-f1-border shadow-card'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 cursor-pointer group">
          <div className="w-8 h-8 bg-f1-red rounded flex items-center justify-center group-hover:scale-105 transition-transform">
            <span className="font-display font-black text-white text-sm">F1</span>
          </div>
          <span className="font-display font-bold text-lg tracking-tight text-f1-light">
            Strategy<span className="text-f1-red">Optimizer</span>
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {/* Page tabs */}
          <div className="flex items-center bg-f1-card border border-f1-border rounded-lg p-0.5 mr-4">
            <Link
              href="/"
              className={`text-sm font-mono px-4 py-1.5 rounded-md transition-all ${
                !isSimulate
                  ? 'bg-f1-red text-white shadow-sm'
                  : 'text-f1-muted hover:text-f1-light'
              }`}
            >
              Analysis
            </Link>
            <Link
              href="/simulate"
              className={`text-sm font-mono px-4 py-1.5 rounded-md transition-all ${
                isSimulate
                  ? 'bg-emerald-500 text-white shadow-sm'
                  : 'text-f1-muted hover:text-f1-light'
              }`}
            >
              Race Simulator
            </Link>
          </div>

          {/* Section nav (only on main page) */}
          {!isSimulate && (
            <>
              {[
                ['Circuits', 'circuits'],
                ['Strategy', 'strategy'],
                ['RL Agent', 'rl'],
                ['Validation', 'validation'],
                ['Backtest', 'backtest'],
                ['Sensitivity', 'sensitivity'],
              ].map(([label, id]) => (
                <button
                  key={id}
                  onClick={() => scrollTo(id)}
                  className="text-sm font-body text-f1-muted hover:text-f1-light transition-colors px-2 py-1"
                >
                  {label}
                </button>
              ))}
            </>
          )}

          <div className="flex items-center gap-2 ml-3">
            <ThemeToggle />
            <a
              href="https://github.com/ImSounic/f1-strategy-prediction"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-mono px-4 py-1.5 border border-f1-border rounded-lg hover:border-f1-red hover:text-f1-red text-f1-muted transition-all"
            >
              GitHub
            </a>
          </div>
        </nav>

        {/* Mobile: just theme toggle */}
        <div className="md:hidden">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
