'use client'

import { useState } from 'react'
import { sensitivityData, type SensitivityParam } from '@/data/sensitivityResults'
import { useChartTheme } from '@/lib/ThemeProvider'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

const CATEGORY_COLORS: Record<string, string> = {
  pit: '#f59e0b',
  safety_car: '#ef4444',
  tyre: '#22c55e',
  fuel: '#3b82f6',
  noise: '#a855f7',
}

const CATEGORY_LABELS: Record<string, string> = {
  pit: 'Pit Stop',
  safety_car: 'Safety Car',
  tyre: 'Tyre Wear',
  fuel: 'Fuel',
  noise: 'Noise',
}

function StabilityBadge({ param }: { param: SensitivityParam }) {
  const changes = param.levels.filter(l => l.topChanged).length
  const total = param.levels.length

  if (changes === 0) {
    return (
      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-green-500/20 text-green-400 border border-green-500/30">
        STABLE
      </span>
    )
  }
  if (changes <= 2) {
    return (
      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
        SENSITIVE ({changes}/{total})
      </span>
    )
  }
  return (
    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-red-500/20 text-red-400 border border-red-500/30">
      VOLATILE ({changes}/{total})
    </span>
  )
}

export function SensitivityDashboard() {
  const chart = useChartTheme()
  const circuits = Object.keys(sensitivityData)
  const [selectedCircuit, setSelectedCircuit] = useState(circuits[0] || '')

  const data = sensitivityData[selectedCircuit]
  if (!data) return null

  // Summary: how many params cause top strategy changes, across all circuits
  const summaryData = data.parameters.map(p => ({
    name: p.label.length > 18 ? p.label.slice(0, 16) + '...' : p.label,
    fullName: p.label,
    changes: p.levels.filter(l => l.topChanged).length,
    maxSwaps: Math.max(...p.levels.map(l => l.top5RankSwaps)),
    category: p.category,
    color: CATEGORY_COLORS[p.category] || '#888',
  }))

  // Cross-circuit sensitivity heatmap data
  const heatmapData = Object.entries(sensitivityData).map(([ck, cd]) => {
    const row: Record<string, number | string> = { circuit: cd.circuitName.replace(' Grand Prix', '') }
    for (const p of cd.parameters) {
      row[p.key] = p.levels.filter(l => l.topChanged).length
    }
    return row
  })

  return (
    <section id="sensitivity" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Sensitivity Analysis
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Parameter Robustness
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            How stable are strategy rankings when we perturb key simulation parameters?
            Each parameter is tested at 5 levels (±10-50% of default).
            <span className="text-green-400"> Stable</span> = top strategy unchanged,{' '}
            <span className="text-red-400">Volatile</span> = top strategy changes frequently.
          </p>
        </div>

        {/* Circuit selector */}
        <div className="flex flex-wrap gap-2 mb-8">
          {circuits.map(ck => (
            <button
              key={ck}
              onClick={() => setSelectedCircuit(ck)}
              className={`px-3 py-1.5 rounded-lg font-mono text-xs transition-colors ${
                selectedCircuit === ck
                  ? 'bg-f1-red text-white'
                  : 'theme-card text-f1-muted hover:text-f1-light'
              }`}
            >
              {sensitivityData[ck].circuitName.replace(' Grand Prix', '')}
            </button>
          ))}
        </div>

        {/* Baseline info */}
        <div className="theme-card rounded-xl p-4 mb-8 flex items-center gap-4">
          <div>
            <div className="font-mono text-xs text-f1-muted uppercase">Baseline Strategy</div>
            <div className="font-display font-bold text-lg text-f1-light">{data.baselineTop}</div>
          </div>
          <div className="ml-auto font-mono text-xs text-f1-muted">
            {data.nSims} sims/level &middot; {data.parameters.length} params &middot; 5 levels each
          </div>
        </div>

        {/* Impact chart */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          <div className="theme-card rounded-xl p-6">
            <h4 className="font-display font-bold text-lg mb-6">Strategy Changes per Parameter</h4>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={summaryData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: chart.grid }}
                  tickLine={false}
                  domain={[0, 5]}
                  allowDecimals={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: chart.text, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                  width={130}
                />
                <Tooltip
                  contentStyle={{
                    background: chart.tooltipBg,
                    border: `1px solid ${chart.tooltipBorder}`,
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '12px',
                  }}
                  itemStyle={{ color: chart.tooltipText }}
                  labelStyle={{ color: chart.tooltipText }}
                  formatter={(value: number) => [
                    `${value} out of 5 levels`,
                    'Top strategy changes',
                  ]}
                />
                <Bar dataKey="changes" radius={[0, 4, 4, 0]}>
                  {summaryData.map((d, i) => (
                    <Cell key={i} fill={d.changes === 0 ? '#22c55e' : d.changes <= 2 ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Cross-circuit heatmap */}
          {heatmapData.length > 1 && (
            <div className="theme-card rounded-xl p-6">
              <h4 className="font-display font-bold text-lg mb-6">Cross-Circuit Sensitivity</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-f1-border">
                      <th className="text-left py-2 px-1 font-mono text-f1-muted">Circuit</th>
                      {data.parameters.map(p => (
                        <th key={p.key} className="text-center py-2 px-1 font-mono text-f1-muted" title={p.label}>
                          {p.label.split(' ')[0].slice(0, 6)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {heatmapData.map((row, i) => (
                      <tr key={i} className="border-b border-f1-border/30">
                        <td className="py-2 px-1 font-mono text-f1-light">{row.circuit as string}</td>
                        {data.parameters.map(p => {
                          const val = row[p.key] as number
                          const bg = val === 0 ? '#22c55e20' : val <= 2 ? '#f59e0b20' : '#ef444420'
                          const color = val === 0 ? '#22c55e' : val <= 2 ? '#f59e0b' : '#ef4444'
                          return (
                            <td key={p.key} className="text-center py-2 px-1">
                              <span
                                className="inline-block w-6 h-6 leading-6 rounded font-mono font-bold"
                                style={{ backgroundColor: bg, color }}
                              >
                                {val}
                              </span>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex gap-4 mt-4 justify-center">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm bg-green-500/30" />
                  <span className="font-mono text-[10px] text-f1-muted">0 changes</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm bg-amber-500/30" />
                  <span className="font-mono text-[10px] text-f1-muted">1-2 changes</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm bg-red-500/30" />
                  <span className="font-mono text-[10px] text-f1-muted">3+ changes</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Detailed parameter cards */}
        <div className="space-y-4">
          <h4 className="font-display font-bold text-lg">Parameter Details</h4>
          {data.parameters.map(param => (
            <div key={param.key} className="theme-card rounded-xl p-5">
              <div className="flex items-center gap-3 mb-3">
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase"
                  style={{
                    backgroundColor: `${CATEGORY_COLORS[param.category]}20`,
                    color: CATEGORY_COLORS[param.category],
                  }}
                >
                  {CATEGORY_LABELS[param.category]}
                </span>
                <h5 className="font-display font-bold text-f1-light">{param.label}</h5>
                <span className="font-mono text-xs text-f1-muted">
                  default: {param.default}{param.unit}
                </span>
                <div className="ml-auto">
                  <StabilityBadge param={param} />
                </div>
              </div>

              <div className="font-body text-xs text-f1-muted mb-4 max-w-2xl">{param.source}</div>

              <div className="grid grid-cols-5 gap-2">
                {param.levels.map((level, i) => (
                  <div
                    key={i}
                    className={`rounded-lg p-3 border transition-colors ${
                      level.topChanged
                        ? 'border-red-500/50 bg-red-500/5'
                        : level.multiplier === 1 ? 'border-f1-red/30 bg-f1-red/5' : 'border-f1-border bg-f1-dark/30'
                    }`}
                  >
                    <div className="font-mono text-[10px] text-f1-muted uppercase mb-1">
                      {level.label}
                    </div>
                    <div className="font-mono text-xs text-f1-light font-bold mb-1">
                      {level.value}{param.unit}
                    </div>
                    <div className={`font-mono text-[10px] truncate ${level.topChanged ? 'text-red-400' : 'text-f1-muted'}`}>
                      {level.topStrategy.split('(')[0].trim().replace(/^\d-stop /, '')}
                    </div>
                    {level.top5RankSwaps > 0 && (
                      <div className="font-mono text-[10px] text-amber-400 mt-1">
                        {level.top5RankSwaps} rank swaps
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Methodology note */}
        <div className="mt-8 p-6 rounded-xl border border-f1-border/50">
          <h4 className="font-display font-bold text-sm mb-3 text-f1-muted uppercase tracking-wider">
            Methodology
          </h4>
          <div className="font-body text-sm text-f1-muted leading-relaxed space-y-2">
            <p>
              Each parameter is tested at 5 levels (typically ±10-50% of default).
              At each level, all {data.parameters[0]?.levels[0]?.top5.length ? 'strategies' : '~78 strategies'} are
              re-simulated with {data.nSims} Monte Carlo runs and re-ranked by median race time.
            </p>
            <p>
              <strong className="text-f1-light">Key finding:</strong> Safety Car probability is the most
              volatile parameter — small changes in SC likelihood can flip the optimal strategy entirely.
              Most other parameters (fuel, degradation, pit loss) are robust within ±20%.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
