'use client'

import { useRef } from 'react'
import { circuits } from '@/data/circuits'
import { strategyResults } from '@/data/strategies'

interface Props {
  selected: string
  onSelect: (key: string) => void
}

const charLabels: Record<string, string> = {
  abrasiveness: 'Abrasiveness',
  grip: 'Grip Level',
  traction: 'Traction',
  braking: 'Braking',
  lateral: 'Lateral',
  stress: 'Tyre Stress',
  downforce: 'Downforce',
  evolution: 'Track Evo',
}

const compoundColors: Record<string, { bg: string; text: string }> = {
  C1: { bg: 'bg-gray-300', text: 'text-gray-900' },
  C2: { bg: 'bg-gray-300', text: 'text-gray-900' },
  C3: { bg: 'bg-yellow-400', text: 'text-gray-900' },
  C4: { bg: 'bg-red-500', text: 'text-white' },
  C5: { bg: 'bg-red-500', text: 'text-white' },
}

function CompoundBadge({ compound }: { compound: string }) {
  const info = compoundColors[compound] || { bg: 'bg-f1-border', text: 'text-white' }
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${info.bg} ${info.text}`}>
      {compound}
    </span>
  )
}

function CompoundWithRole({ compound, role }: { compound: string; role: string }) {
  const colorMap: Record<string, string> = {
    HARD: 'bg-gray-300 text-gray-900 border-gray-400',
    MEDIUM: 'bg-yellow-400 text-gray-900 border-yellow-500',
    SOFT: 'bg-red-500 text-white border-red-600',
  }
  const cls = colorMap[role] || 'bg-f1-border text-white border-f1-border'
  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className={`inline-flex items-center justify-center w-11 h-11 rounded-full text-xs font-mono font-bold border-2 ${cls}`}>
        {compound}
      </span>
      <span className="font-mono text-[10px] text-f1-muted uppercase tracking-wider">{role}</span>
    </div>
  )
}

export function CircuitExplorer({ selected, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const circuit = circuits.find(c => c.key === selected) || circuits[0]
  const compounds = circuit.compounds.split('/')
  const roles = ['HARD', 'MEDIUM', 'SOFT']

  const hasData = (key: string) => {
    return Object.keys(strategyResults).some(k => k.startsWith(key + '_'))
  }

  const scroll = (dir: 'left' | 'right') => {
    if (!scrollRef.current) return
    const amount = 300
    scrollRef.current.scrollBy({ left: dir === 'left' ? -amount : amount, behavior: 'smooth' })
  }

  return (
    <section id="circuits" className="py-20 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section header */}
        <div className="mb-8">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Circuit Analysis
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight text-f1-light">
            Select a Circuit
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-xl">
            Each circuit has unique characteristics that determine optimal tyre strategy.
            Select a circuit to view its Monte Carlo strategy recommendations.
          </p>
        </div>

        {/* Horizontal scrollable circuit carousel */}
        <div className="relative mb-10">
          {/* Scroll buttons - outside the scroll area */}
          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={() => scroll('left')}
              className="w-9 h-9 bg-f1-card border border-f1-border rounded-full flex items-center justify-center text-f1-muted hover:text-f1-red hover:border-f1-red transition-all shadow-card flex-shrink-0"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6"/></svg>
            </button>
            <button
              onClick={() => scroll('right')}
              className="w-9 h-9 bg-f1-card border border-f1-border rounded-full flex items-center justify-center text-f1-muted hover:text-f1-red hover:border-f1-red transition-all shadow-card flex-shrink-0"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>
            </button>
            <span className="font-mono text-xs text-f1-muted ml-1">Scroll to browse circuits</span>
          </div>

          {/* Carousel track */}
          <div
            ref={scrollRef}
            className="circuit-carousel flex gap-3 overflow-x-auto py-2"
          >
            {circuits.map(c => {
              const isSelected = selected === c.key
              const hasResults = hasData(c.key)
              return (
                <button
                  key={c.key}
                  onClick={() => onSelect(c.key)}
                  className={`flex-shrink-0 group rounded-xl transition-all text-left relative overflow-hidden ${
                    isSelected
                      ? 'bg-f1-red/10 border-2 border-f1-red shadow-card-hover'
                      : 'bg-f1-card border border-f1-border hover:border-f1-muted shadow-card hover:shadow-card-hover'
                  }`}
                  style={{ width: '160px' }}
                >
                  <div className="p-4">
                    {hasResults && (
                      <div className="absolute top-3 right-3">
                        <span className="w-2 h-2 rounded-full bg-green-500 block" />
                      </div>
                    )}
                    <div className="text-2xl mb-2">{c.country}</div>
                    <div className={`font-display font-bold text-sm leading-tight transition-colors ${
                      isSelected ? 'text-f1-light' : 'text-f1-muted group-hover:text-f1-light'
                    }`}>
                      {c.name.length > 18 ? c.name.split(' ').slice(0, 2).join(' ') : c.name}
                    </div>
                    <div className="font-mono text-[11px] text-f1-muted mt-1.5">
                      {c.totalLaps} laps
                    </div>
                    <div className="flex gap-1 mt-2.5">
                      {c.compounds.split('/').map((comp, i) => (
                        <CompoundBadge key={i} compound={comp} />
                      ))}
                    </div>
                  </div>
                  {/* Bottom accent for selected */}
                  {isSelected && (
                    <div className="h-0.5 bg-f1-red" />
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Circuit detail card */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Left: Info */}
          <div className="theme-card rounded-xl p-8 racing-stripe">
            <h3 className="font-display font-bold text-2xl pl-4 mb-6 text-f1-light">
              {circuit.name}
            </h3>

            <div className="grid grid-cols-2 gap-4 pl-4 mb-6">
              {[
                { label: 'Total Laps', value: circuit.totalLaps.toString() },
                { label: 'Pit Loss', value: `${circuit.pitLoss}s` },
                { label: 'SC Probability', value: `${(circuit.scProbability * 100).toFixed(0)}%` },
                { label: 'Pit Window', value: circuit.totalLaps > 60 ? 'Wide' : 'Tight' },
              ].map((item, i) => (
                <div key={i}>
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">
                    {item.label}
                  </div>
                  <div className="font-display font-bold text-xl text-f1-light mt-1">
                    {item.value}
                  </div>
                </div>
              ))}
            </div>

            <div className="pl-4 pt-4 border-t border-f1-border">
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-4">
                Tyre Allocation
              </div>
              <div className="flex gap-6 items-end">
                {compounds.map((comp, i) => (
                  <CompoundWithRole key={i} compound={comp} role={roles[i] || 'MEDIUM'} />
                ))}
              </div>
            </div>
          </div>

          {/* Right: Characteristics */}
          <div className="theme-card rounded-xl p-8">
            <h4 className="font-display font-bold text-lg mb-6 text-f1-light">
              Pirelli Characteristics
            </h4>

            <div className="space-y-3">
              {Object.entries(circuit.characteristics).map(([key, val]) => {
                const pct = (val / 5) * 100
                const isHigh = val >= 4
                const isCritical = val >= 5
                return (
                  <div key={key} className="flex items-center gap-4">
                    <div className="w-24 font-mono text-xs text-f1-muted">
                      {charLabels[key] || key}
                    </div>
                    <div className="flex-1 h-2 bg-f1-darker rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isCritical ? 'bg-f1-red' : isHigh ? 'bg-orange-500' : 'bg-f1-red/60'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="flex gap-0.5">
                      {[1, 2, 3, 4, 5].map(n => (
                        <div
                          key={n}
                          className={`w-1.5 h-3 rounded-sm transition-colors ${
                            n <= val
                              ? val >= 4 ? 'bg-f1-red' : 'bg-f1-red/60'
                              : 'bg-f1-darker'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="mt-6 pt-4 border-t border-f1-border">
              <div className="font-body text-xs text-f1-muted leading-relaxed">
                {circuit.characteristics.abrasiveness >= 3
                  ? 'High abrasiveness track — expect significant tyre degradation, favouring multi-stop strategies.'
                  : circuit.characteristics.downforce >= 4
                  ? 'High downforce circuit — mechanical grip matters less, aero-limited corners dominate strategy.'
                  : circuit.characteristics.evolution >= 4
                  ? 'High track evolution — rubber build-up improves grip throughout the weekend, late stints benefit.'
                  : 'Balanced circuit characteristics — strategy flexibility allows varied approaches.'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
