'use client'

import { useState } from 'react'
import { Header } from '@/components/Header'
import { Hero } from '@/components/Hero'
import { Stats } from '@/components/Stats'
import { Methodology } from '@/components/Methodology'
import { CircuitExplorer } from '@/components/CircuitExplorer'
import { StrategyView } from '@/components/StrategyView'
import { ScenarioView } from '@/components/ScenarioView'
import { RLView } from '@/components/RLView'
import { ValidationDashboard } from '@/components/ValidationDashboard'
import { Limitations } from '@/components/Limitations'
import { TechStack } from '@/components/TechStack'
import { Footer } from '@/components/Footer'

export default function Home() {
  const [selectedCircuit, setSelectedCircuit] = useState('bahrain')

  return (
    <main className="min-h-screen">
      <Header />
      <Hero />
      <Stats />
      <Methodology />
      <CircuitExplorer
        selected={selectedCircuit}
        onSelect={setSelectedCircuit}
      />
      <StrategyView circuitKey={selectedCircuit} />
      <ScenarioView circuitKey={selectedCircuit} />
      <RLView circuitKey={selectedCircuit} />
      <ValidationDashboard />
      <Limitations />
      <TechStack />
      <Footer />
    </main>
  )
}
