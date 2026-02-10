'use client'

import { circuits } from '@/data/circuits'

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

export function CircuitExplorer({ selected, onSelect }: Props) {
  const circuit = circuits.find(c => c.key === selected) || circuits[0]

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
            Data sourced from Pirelli circuit infographics.
          </p>
        </div>

        {/* Circuit grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-12">
          {circuits.map(c => (
            <button
              key={c.key}
              onClick={() => onSelect(c.key)}
              className={`group p-4 rounded border transition-all text-left ${
                selected === c.key
                  ? 'border-f1-red bg-f1-red/10'
                  : 'border-f1-border bg-f1-card hover:border-f1-muted'
              }`}
            >
              <div className="font-mono text-lg mb-1">{c.country}</div>
              <div className={`font-display font-bold text-sm leading-tight ${
                selected === c.key ? 'text-white' : 'text-f1-muted group-hover:text-white'
              } transition-colors`}>
                {c.name.split(' ').slice(0, 2).join(' ')}
              </div>
              <div className="font-mono text-xs text-f1-border mt-2">
                {c.totalLaps} laps
              </div>
            </button>
          ))}
        </div>

        {/* Circuit detail card */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Left: Info */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-8 racing-stripe">
            <h3 className="font-display font-bold text-2xl pl-4 mb-6">
              {circuit.name}
            </h3>

            <div className="grid grid-cols-2 gap-4 pl-4">
              {[
                { label: 'Total Laps', value: circuit.totalLaps.toString() },
                { label: 'Pit Loss', value: `${circuit.pitLoss}s` },
                { label: 'SC Probability', value: `${(circuit.scProbability * 100).toFixed(0)}%` },
                { label: 'Compounds', value: circuit.compounds },
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
          </div>

          {/* Right: Characteristics radar */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-8">
            <h4 className="font-display font-bold text-lg mb-6">
              Pirelli Characteristics
            </h4>

            <div className="space-y-3">
              {Object.entries(circuit.characteristics).map(([key, val]) => (
                <div key={key} className="flex items-center gap-4">
                  <div className="w-24 font-mono text-xs text-f1-muted">
                    {charLabels[key] || key}
                  </div>
                  <div className="flex-1 h-2 bg-f1-darker rounded-full overflow-hidden">
                    <div
                      className="h-full bg-f1-red rounded-full transition-all duration-500"
                      style={{ width: `${(val / 5) * 100}%` }}
                    />
                  </div>
                  <div className="w-6 font-mono text-sm text-f1-muted text-right">
                    {val}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
