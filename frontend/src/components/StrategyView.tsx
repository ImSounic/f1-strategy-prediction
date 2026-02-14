'use client'

import { useState, useEffect } from 'react'
import { strategyResults, StrategyResult } from '@/data/strategies'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

interface Props {
  circuitKey: string
}

const compoundColorMap: Record<string, string> = {
  S: '#E10600',
  M: '#FFD700',
  H: '#FFFFFF',
}

function CompoundPill({ compound }: { compound: string }) {
  // Normalise: accept full names or single letters
  const c = compound.trim()
  const key = c === 'SOFT' || c === 'S' ? 'S'
            : c === 'MEDIUM' || c === 'M' ? 'M'
            : c === 'HARD' || c === 'H' ? 'H'
            : c[0] || '?'

  const label = c === 'SOFT' || c === 'S' ? 'SOFT'
              : c === 'MEDIUM' || c === 'M' ? 'MEDIUM'
              : c === 'HARD' || c === 'H' ? 'HARD'
              : c

  const styles: Record<string, { border: string; text: string }> = {
    S: { border: 'border-red-500 text-red-400', text: 'text-red-400' },
    M: { border: 'border-yellow-400 text-yellow-300', text: 'text-yellow-300' },
    H: { border: 'border-white text-white', text: 'text-white' },
  }
  const s = styles[key] || { border: 'border-f1-border text-f1-muted', text: 'text-f1-muted' }

  return (
    <span
      className={`inline-block px-2 py-0.5 rounded border text-[10px] font-mono font-bold leading-tight ${s.border}`}
    >
      {label}
    </span>
  )
}

function CompoundSequence({ compounds }: { compounds: string }) {
  // Parse "HARD → MEDIUM → SOFT" or "M→H" or "H → M → S" format
  const parts = compounds.split(/[→>]/).map(c => c.trim()).filter(Boolean)
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {parts.map((comp, i) => (
        <div key={i} className="flex items-center gap-1">
          <CompoundPill compound={comp} />
        </div>
      ))}
    </div>
  )
}

function StintTimeline({ strategy }: { strategy: StrategyResult }) {
  if (!strategy.stintLengths || strategy.stintLengths.length === 0) return null

  const compounds = strategy.compounds.split(/\s*→\s*/)
  const total = strategy.stintLengths.reduce((a, b) => a + b, 0)

  const compoundBg: Record<string, string> = {
    SOFT: 'bg-red-500',
    MEDIUM: 'bg-yellow-400',
    HARD: 'bg-white',
  }
  const compoundText: Record<string, string> = {
    SOFT: 'text-white',
    MEDIUM: 'text-gray-900',
    HARD: 'text-gray-900',
  }

  return (
    <div className="flex h-6 rounded overflow-hidden w-full mt-3">
      {strategy.stintLengths.map((len, i) => {
        const compound = compounds[i]?.trim() || ''
        return (
          <div
            key={i}
            className={`${compoundBg[compound] || 'bg-f1-border'} relative flex items-center justify-center`}
            style={{ width: `${(len / total) * 100}%` }}
          >
            <span className={`text-[10px] font-mono font-bold ${compoundText[compound] || 'text-white'} opacity-90`}>
              {len}L
            </span>
            {i < strategy.stintLengths.length - 1 && (
              <div className="absolute right-0 top-0 bottom-0 w-0.5 bg-f1-darker" />
            )}
          </div>
        )
      })}
    </div>
  )
}

export function StrategyView({ circuitKey }: Props) {
  const matchingKey = Object.keys(strategyResults)
    .find(k => k.startsWith(circuitKey + '_')) || ''

  const [selectedKey, setSelectedKey] = useState(matchingKey)

  useEffect(() => {
    const key = Object.keys(strategyResults)
      .find(k => k.startsWith(circuitKey + '_')) || ''
    setSelectedKey(key)
  }, [circuitKey])

  const data = selectedKey ? strategyResults[selectedKey] : null

  if (!data || !data.strategies.length) {
    return (
      <section id="strategy" className="py-24 border-b border-f1-border">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-12">
            <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
              Strategy Simulation
            </div>
            <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
              Monte Carlo Results
            </h2>
          </div>
          <div className="bg-f1-card border border-f1-border rounded-lg p-12 text-center">
            <div className="text-4xl mb-4">🏎️</div>
            <div className="font-display text-xl text-f1-muted mb-2">
              No pre-computed results for this circuit
            </div>
            <p className="font-body text-sm text-f1-border max-w-md mx-auto">
              Run the precompute script locally to generate Monte Carlo results
              for all circuits, then redeploy.
            </p>
          </div>
        </div>
      </section>
    )
  }

  const strategies = data.strategies
  const best = strategies[0]

  const chartData = strategies.slice(0, 8).map(s => ({
    name: s.cleanName,
    delta: s.delta,
    stops: s.stops,
  }))

  const gap = strategies.length > 1 ? strategies[1].delta : 999
  const confidence = gap < 5 ? 'LOW' : gap < 15 ? 'MEDIUM' : 'HIGH'
  const confColor = confidence === 'HIGH'
    ? 'text-green-400'
    : confidence === 'MEDIUM'
      ? 'text-yellow-400'
      : 'text-orange-400'
  const confDesc = confidence === 'HIGH'
    ? 'Clear optimal strategy with significant gap to alternatives'
    : confidence === 'MEDIUM'
      ? 'Marginal advantage — safety car timing could swing results'
      : 'Multiple viable strategies — track position may decide'

  const fmt = (n: number) => new Intl.NumberFormat('en-US').format(n)

  return (
    <section id="strategy" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Strategy Simulation
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Monte Carlo Results
          </h2>
          <p className="font-body text-f1-muted mt-3">
            {data.circuitName} {data.season} &mdash; {fmt(data.nSims)} simulations per strategy with stochastic safety car injection.
          </p>
        </div>

        {/* Recommendation banner */}
        <div className="bg-f1-card border border-f1-border rounded-lg p-6 mb-8 racing-stripe">
          <div className="pl-4 flex items-start justify-between flex-wrap gap-6">
            <div>
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-1">
                Recommended Strategy
              </div>
              <div className="font-display font-black text-3xl text-white mb-2">
                {best.cleanName}
              </div>
              <div className="flex items-center gap-3">
                <CompoundSequence compounds={best.compounds.replace(/→/g, '→')} />
                <span className="font-mono text-xs text-f1-border">{best.stops}-stop</span>
              </div>
              {best.pitLaps && best.pitLaps.length > 0 && (
                <div className="font-mono text-xs text-f1-muted mt-2">
                  Pit on Lap {best.pitLaps.join(', ')}
                </div>
              )}
              <StintTimeline strategy={best} />
            </div>
            <div className="text-right">
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-1">
                Confidence
              </div>
              <div className={`font-display font-bold text-2xl ${confColor}`}>
                {confidence}
              </div>
              <div className="font-mono text-xs text-f1-border mt-1">
                +{gap.toFixed(1)}s gap to P2
              </div>
              <div className="font-body text-xs text-f1-border mt-2 max-w-[200px] text-right">
                {confDesc}
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Delta bar chart */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h4 className="font-display font-bold text-lg mb-6">
              Strategy Rankings &mdash; Delta to Best
            </h4>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 90, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A3A" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: '#8A8A9A', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#2A2A3A' }}
                  tickLine={false}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fill: '#E8E8F0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                  width={85}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1A1A26',
                    border: '1px solid #2A2A3A',
                    borderRadius: '4px',
                    fontFamily: 'JetBrains Mono',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`+${value.toFixed(1)}s`, 'Delta']}
                />
                <Bar dataKey="delta" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.stops === 1 ? '#E10600' : '#0090D0'}
                      opacity={i === 0 ? 1 : 0.7}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-6 mt-4 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-[#E10600]" />
                <span className="font-mono text-xs text-f1-muted">1-stop</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-[#0090D0]" />
                <span className="font-mono text-xs text-f1-muted">2-stop</span>
              </div>
            </div>
          </div>

          {/* Confidence intervals */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h4 className="font-display font-bold text-lg mb-6">
              Confidence Intervals (P5&ndash;P95)
            </h4>
            <div className="space-y-6 mt-8">
              {strategies.slice(0, 6).map((s, i) => {
                const allP5 = strategies.slice(0, 6).map(x => x.p5)
                const allP95 = strategies.slice(0, 6).map(x => x.p95)
                const minP5 = Math.min(...allP5)
                const maxP95 = Math.max(...allP95)
                const range = maxP95 - minP5 || 1
                const leftPct = ((s.p5 - minP5) / range) * 100
                const widthPct = ((s.p95 - s.p5) / range) * 100
                const medianPct = ((s.medianTime - minP5) / range) * 100

                return (
                  <div key={i}>
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center gap-2">
                        <span className={`font-display font-bold text-xs ${i === 0 ? 'text-f1-red' : 'text-f1-muted'}`}>
                          P{i + 1}
                        </span>
                        <span className="font-mono text-xs text-f1-muted">{s.cleanName}</span>
                      </div>
                      <span className="font-mono text-xs text-f1-border">
                        {s.p5.toFixed(0)}s &ndash; {s.p95.toFixed(0)}s
                      </span>
                    </div>
                    <div className="relative h-4 bg-f1-darker rounded-full">
                      <div
                        className="absolute h-full bg-blue-500/30 rounded-full"
                        style={{ left: `${leftPct}%`, width: `${Math.max(widthPct, 2)}%` }}
                      />
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-f1-red rounded-full border-2 border-white"
                        style={{ left: `${Math.min(Math.max(medianPct, 2), 98)}%`, marginLeft: '-6px' }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="flex gap-6 mt-8 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-f1-red border-2 border-white" />
                <span className="font-mono text-xs text-f1-muted">Median</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-3 rounded-full bg-blue-500/30" />
                <span className="font-mono text-xs text-f1-muted">P5&ndash;P95 range</span>
              </div>
            </div>
          </div>
        </div>

        {/* Strategy table */}
        <div className="bg-f1-card border border-f1-border rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-f1-border">
                {['Rank', 'Strategy', 'Compounds', 'Stops', 'Pit Laps', 'Median', 'Delta', 'Std Dev', 'SC Events'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-mono text-xs text-f1-muted uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {strategies.map((s, i) => (
                <tr
                  key={i}
                  className={`border-b border-f1-border/50 transition-colors hover:bg-f1-darker/50 ${
                    i === 0 ? 'bg-f1-red/5' : ''
                  }`}
                >
                  <td className="px-4 py-3">
                    <span className={`font-display font-bold ${i === 0 ? 'text-f1-red' : 'text-f1-muted'}`}>
                      P{i + 1}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="font-display font-bold text-sm">{s.cleanName}</div>
                  </td>
                  <td className="px-4 py-3">
                    <CompoundSequence compounds={s.compounds} />
                  </td>
                  <td className="px-4 py-3 font-mono text-sm">{s.stops}</td>
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-f1-muted">
                      {s.pitLaps && s.pitLaps.length > 0
                        ? s.pitLaps.map(lap => `L${lap}`).join(', ')
                        : '-'}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-sm">{s.medianTime.toFixed(1)}s</td>
                  <td className="px-4 py-3">
                    <span className={`font-mono text-sm ${i === 0 ? 'text-green-400 font-bold' : 'text-f1-muted'}`}>
                      {i === 0 ? 'BEST' : `+${s.delta.toFixed(1)}s`}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-sm text-f1-muted">&plusmn;{s.stdTime.toFixed(1)}</td>
                  <td className="px-4 py-3 font-mono text-sm text-f1-muted">{s.scEvents.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
