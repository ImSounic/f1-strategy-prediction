'use client'

import { useState, useEffect } from 'react'
import { Header } from '@/components/Header'
import { Hero } from '@/components/Hero'
import { Stats } from '@/components/Stats'
import { CircuitExplorer } from '@/components/CircuitExplorer'
import { StrategyView } from '@/components/StrategyView'
import { ValidationDashboard } from '@/components/ValidationDashboard'
import { TechStack } from '@/components/TechStack'
import { Footer } from '@/components/Footer'
import { healthCheck } from '@/lib/api'

export default function Home() {
  const [selectedCircuit, setSelectedCircuit] = useState('bahrain')
  const [apiOnline, setApiOnline] = useState(false)

  useEffect(() => {
    healthCheck().then(setApiOnline)
    // Re-check every 30s
    const interval = setInterval(() => {
      healthCheck().then(setApiOnline)
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <main className="min-h-screen">
      <Header />
      <Hero />
      <Stats />
      <CircuitExplorer
        selected={selectedCircuit}
        onSelect={setSelectedCircuit}
      />
      <StrategyView circuitKey={selectedCircuit} apiOnline={apiOnline} />
      <ValidationDashboard />
      <TechStack />
      <Footer />
    </main>
  )
}
