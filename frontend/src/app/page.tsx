'use client'

import { useState } from 'react'
import { Header } from '@/components/Header'
import { Hero } from '@/components/Hero'
import { Stats } from '@/components/Stats'
import { CircuitExplorer } from '@/components/CircuitExplorer'
import { StrategyView } from '@/components/StrategyView'
import { ValidationDashboard } from '@/components/ValidationDashboard'
import { TechStack } from '@/components/TechStack'
import { Footer } from '@/components/Footer'

export default function Home() {
  const [selectedCircuit, setSelectedCircuit] = useState('bahrain')

  return (
    <main className="min-h-screen">
      <Header />
      <Hero />
      <Stats />
      <CircuitExplorer
        selected={selectedCircuit}
        onSelect={setSelectedCircuit}
      />
      <StrategyView circuitKey={selectedCircuit} />
      <ValidationDashboard />
      <TechStack />
      <Footer />
    </main>
  )
}
