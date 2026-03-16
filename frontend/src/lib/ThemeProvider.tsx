'use client'

import { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  toggle: () => void
  isDark: boolean
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  toggle: () => {},
  isDark: true,
})

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const stored = localStorage.getItem('f1-theme') as Theme | null
    if (stored) {
      setTheme(stored)
    } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
      setTheme('light')
    }
  }, [])

  useEffect(() => {
    if (!mounted) return
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    localStorage.setItem('f1-theme', theme)
  }, [theme, mounted])

  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <ThemeContext.Provider value={{ theme, toggle, isDark: theme === 'dark' }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => useContext(ThemeContext)

export function useChartTheme() {
  const { isDark } = useTheme()
  return {
    grid: isDark ? '#252538' : '#e0e0ea',
    axis: isDark ? '#252538' : '#e0e0ea',
    text: isDark ? '#7a7a95' : '#5a5a72',
    tooltipBg: isDark ? '#141422' : '#ffffff',
    tooltipBorder: isDark ? '#252538' : '#d8d8e5',
    tooltipText: isDark ? '#eeeef5' : '#12121e',
    red: '#E10600',
    blue: '#0090D0',
    green: '#10b981',
    amber: '#f59e0b',
    purple: '#8b5cf6',
  }
}
