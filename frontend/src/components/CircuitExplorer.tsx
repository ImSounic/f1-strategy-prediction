'use client'

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

const compoundColors: Record<string, { bg: string; text: string; label: string }> = {
  C1: { bg: 'bg-white', text: 'text-gray-900', label: 'HARD' },
  C2: { bg: 'bg-white', text: 'text-gray-900', label: 'HARD' },
  C3: { bg: 'bg-yellow-400', text: 'text-gray-900', label: 'MEDIUM' },
  C4: { bg: 'bg-red-500', text: 'text-white', label: 'SOFT' },
  C5: { bg: 'bg-red-500', text: 'text-white', label: 'SOFT' },
}

function CompoundBadge({ compound }: { compound: string }) {
  const info = compoundColors[compound] || { bg: 'bg-f1-border', text: 'text-white', label: compound }
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold ${info.bg} ${info.text}`}>
      {compound}
    </span>
  )
}

function getCompoundRole(compound: string, allCompounds: string[]): string {
  const sorted = [...allCompounds].sort()
  const idx = sorted.indexOf(compound)
  if (idx === 0) return 'HARD'
  if (idx === sorted.length - 1) return 'SOFT'
  return 'MEDIUM'
}

function CompoundWithRole({ compound, role }: { compound: string; role: string }) {
  const colorMap: Record<string, string> = {
    HARD: 'bg-white text-gray-900',
    MEDIUM: 'bg-yellow-400 text-gray-900',
    SOFT: 'bg-red-500 text-white',
  }
  const cls = colorMap[role] || 'bg-f1-border text-white'
  return (
    <div className="flex flex-col items-center gap-1">
      <span className={`inline-flex items-center justify-center w-10 h-10 rounded-full text-xs font-mono font-bold ${cls}`}>
        {compound}
      </span>
      <span className="font-mono text-[10px] text-f1-muted uppercase">{role}</span>
    </div>
  )
}

export function CircuitExplorer({ selected, onSelect }: Props) {
  const circuit = circuits.find(c => c.key === selected) || circuits[0]
  const compounds = circuit.compounds.split('/')
  const roles = ['HARD', 'MEDIUM', 'SOFT']

  // Check if circuit has precomputed data
  const hasData = (key: string) => {
    return Object.keys(strategyResults).some(k => k.startsWith(key + '_'))
  }

  return (
    <section id="circuits" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Circuit Analysis
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Select a Circuit
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-xl">
            Each circuit has unique characteristics that determine optimal tyre strategy.
            Data sourced from Pirelli circuit infographics. Select a circuit to view its
            Monte Carlo strategy recommendations below.
          </p>
        </div>

        {/* Circuit grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-12">
          {circuits.map(c => {
            const isSelected = selected === c.key
            const hasResults = hasData(c.key)
            return (
              <button
                key={c.key}
                onClick={() => onSelect(c.key)}
                className={`group p-4 rounded border transition-all text-left relative ${
                  isSelected
                    ? 'border-f1-red bg-f1-red/10'
                    : 'border-f1-border bg-f1-card hover:border-f1-muted'
                }`}
              >
                {/* Data available indicator */}
                {hasResults && (
                  <div className="absolute top-2 right-2">
                    <span className="w-2 h-2 rounded-full bg-green-500 block" title="Strategy data available" />
                  </div>
                )}
                <div className="font-mono text-lg mb-1">{c.country}</div>
                <div className={`font-display font-bold text-sm leading-tight ${
                  isSelected ? 'text-white' : 'text-f1-muted group-hover:text-white'
                } transition-colors`}>
                  {c.name.length > 25 ? c.name.split(' ').slice(0, 2).join(' ') : c.name}
                </div>
                <div className="font-mono text-xs text-f1-border mt-2">
                  {c.totalLaps} laps
                </div>
                {/* Compound mini badges */}
                <div className="flex gap-1 mt-2">
                  {c.compounds.split('/').map((comp, i) => (
                    <CompoundBadge key={i} compound={comp} />
                  ))}
                </div>
              </button>
            )
          })}
        </div>

        {/* Circuit detail card */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Left: Info */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-8 racing-stripe">
            <h3 className="font-display font-bold text-2xl pl-4 mb-6">
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
                  <div className="font-display font-bold text-xl text-white mt-1">
                    {item.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Compound allocation with F1 colors */}
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
          <div className="bg-f1-card border border-f1-border rounded-lg p-8">
            <h4 className="font-display font-bold text-lg mb-6">
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
                          isCritical ? 'bg-f1-red' : isHigh ? 'bg-orange-500' : 'bg-f1-red/70'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="flex gap-1">
                      {[1, 2, 3, 4, 5].map(n => (
                        <div
                          key={n}
                          className={`w-1.5 h-3 rounded-sm transition-colors ${
                            n <= val
                              ? val >= 4 ? 'bg-f1-red' : 'bg-f1-red/70'
                              : 'bg-f1-darker'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Summary insight */}
            <div className="mt-6 pt-4 border-t border-f1-border">
              <div className="font-body text-xs text-f1-border leading-relaxed">
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
