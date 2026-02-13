'use client'

import { Header } from '@/components/Header'
import { SimulatorView } from '@/components/SimulatorView'
import { Footer } from '@/components/Footer'

export default function SimulatePage() {
  return (
    <main className="min-h-screen">
      <Header />
      <SimulatorView />
      <Footer />
    </main>
  )
}
