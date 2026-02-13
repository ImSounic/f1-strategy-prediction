'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

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
          ? 'bg-f1-darker/95 backdrop-blur-md border-b border-f1-border'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 cursor-pointer">
          <div className="w-8 h-8 bg-f1-red rounded-sm flex items-center justify-center">
            <span className="font-display font-black text-white text-sm">F1</span>
          </div>
          <span className="font-display font-bold text-lg tracking-tight">
            Strategy<span className="text-f1-red">Optimizer</span>
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {/* Page tabs */}
          <div className="flex items-center bg-f1-darker/60 border border-f1-border rounded-lg p-0.5 mr-4">
            <Link
              href="/"
              className={`text-sm font-mono px-4 py-1.5 rounded-md transition-all ${
                !isSimulate
                  ? 'bg-f1-red text-white'
                  : 'text-f1-muted hover:text-white'
              }`}
            >
              Analysis
            </Link>
            <Link
              href="/simulate"
              className={`text-sm font-mono px-4 py-1.5 rounded-md transition-all ${
                isSimulate
                  ? 'bg-emerald-500 text-white'
                  : 'text-f1-muted hover:text-white'
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
              ].map(([label, id]) => (
                <button
                  key={id}
                  onClick={() => scrollTo(id)}
                  className="text-sm font-body text-f1-muted hover:text-white transition-colors px-2"
                >
                  {label}
                </button>
              ))}
            </>
          )}
          
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-mono px-4 py-1.5 border border-f1-border rounded hover:border-f1-red hover:text-f1-red transition-colors ml-3"
          >
            GitHub →
          </a>
        </nav>
      </div>
    </header>
  )
}
