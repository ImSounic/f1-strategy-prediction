'use client'

import { useState } from 'react'
import { simulate, SimulationResponse, ApiStrategyResult } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

interface Props {
  circuitKey: string
  apiOnline: boolean
}

function cleanName(raw: string): string {
  return raw
    .replace(/\s*\(\d+\/\d+(\/\d+)?\)\s*/g, '')
    .replace(/MEDIUM/g, 'M')
    .replace(/HARD/g, 'H')
    .replace(/SOFT/g, 'S')
    .trim()
}

// Deduplicate strategies with same clean name, keep best
function dedup(rankings: ApiStrategyResult[]) {
  const seen = new Map<string, ApiStrategyResult>()
  for (const r of rankings) {
    const key = cleanName(r.strategy_name)
    if (!seen.has(key)) seen.set(key, r)
  }
  return Array.from(seen.values())
}

export function StrategyView({ circuitKey, apiOnline }: Props) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SimulationResponse | null>(null)
  const [nSims, setNSims] = useState(1000)
  const [lastCircuit, setLastCircuit] = useState<string | null>(null)

  // Clear results when circuit changes
  if (circuitKey !== lastCircuit) {
    if (lastCircuit !== null) {
      setResult(null)
      setError(null)
    }
    setLastCircuit(circuitKey)
  }

  const runSimulation = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await simulate(circuitKey, 2025, nSims)
      setResult(data)
    } catch (e: any) {
      setError(e.message || 'Simulation failed')
    } finally {
      setLoading(false)
    }
  }

  // ─── No results yet ───
  if (!result) {
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
            <p className="font-body text-f1-muted mt-3 max-w-xl">
              Run live simulations for any circuit via the FastAPI backend.
              Each strategy is tested across {new Intl.NumberFormat('en-US').format(nSims)} randomized races.
            </p>
          </div>

          <div className="bg-f1-card border border-f1-border rounded-lg p-12 text-center">
            {!apiOnline ? (
              <>
                <div className="flex items-center justify-center gap-2 mb-4">
                  <span className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
                  <span className="font-mono text-sm text-orange-400">API Offline</span>
                </div>
                <div className="font-display text-xl text-f1-muted mb-3">
                  Backend not connected
                </div>
                <p className="font-body text-f1-border max-w-md mx-auto mb-6">
                  Start the FastAPI server locally or deploy to Render to run live simulations.
                </p>
                <code className="inline-block font-mono text-xs text-f1-muted bg-f1-darker px-4 py-2 rounded border border-f1-border">
                  uvicorn src.api.main:app --reload
                </code>
              </>
            ) : (
              <>
                <div className="flex items-center justify-center gap-2 mb-6">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="font-mono text-sm text-green-400">API Connected</span>
                </div>

                {/* Sim count selector */}
                <div className="flex items-center justify-center gap-4 mb-8">
                  <span className="font-mono text-xs text-f1-muted">Simulations:</span>
                  {[500, 1000, 2000, 5000].map(n => (
                    <button
                      key={n}
                      onClick={() => setNSims(n)}
                      className={`px-3 py-1 rounded border font-mono text-xs transition-all ${
                        nSims === n
                          ? 'border-f1-red bg-f1-red/10 text-white'
                          : 'border-f1-border text-f1-muted hover:border-f1-muted'
                      }`}
                    >
                      {new Intl.NumberFormat('en-US').format(n)}
                    </button>
                  ))}
                </div>

                <button
                  onClick={runSimulation}
                  disabled={loading}
                  className="px-10 py-4 bg-f1-red text-white font-display font-bold text-lg uppercase tracking-wider hover:bg-red-700 transition-all glow-red rounded-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <span className="flex items-center gap-3">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Simulating...
                    </span>
                  ) : (
                    <>🏁 Run Simulation</>
                  )}
                </button>

                {error && (
                  <div className="mt-6 font-mono text-sm text-red-400">
                    ✗ {error}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </section>
    )
  }

  // ─── Results view ───
  const strategies = dedup(result.rankings).slice(0, 10)
  const best = strategies[0]
  const bestMedian = best.median_time

  const chartData = strategies.slice(0, 8).map(s => ({
    name: cleanName(s.strategy_name),
    delta: Math.round((s.median_time - bestMedian) * 10) / 10,
    stops: s.num_stops,
  }))

  const gap = strategies.length > 1
    ? Math.round((strategies[1].median_time - bestMedian) * 10) / 10
    : 999
  const confidence = gap < 5 ? 'LOW' : gap < 15 ? 'MEDIUM' : 'HIGH'
  const confColor = confidence === 'HIGH' ? 'text-green-400' : confidence === 'MEDIUM' ? 'text-yellow-400' : 'text-orange-400'

  return (
    <section id="strategy" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="flex flex-wrap items-end justify-between mb-12 gap-4">
          <div>
            <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
              Strategy Simulation
            </div>
            <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
              Monte Carlo Results
            </h2>
            <p className="font-body text-f1-muted mt-3">
              {result.circuit_name} {result.season} — {new Intl.NumberFormat('en-US').format(result.n_sims)} sims × {result.n_strategies} strategies
              in {result.elapsed_seconds.toFixed(1)}s
            </p>
          </div>

          <button
            onClick={runSimulation}
            disabled={loading}
            className="px-6 py-2 border border-f1-border text-f1-muted font-mono text-sm hover:border-f1-red hover:text-f1-red transition-all rounded-sm disabled:opacity-50"
          >
            {loading ? 'Running...' : '↻ Re-simulate'}
          </button>
        </div>

        {/* Recommendation banner */}
        <div className="bg-f1-card border border-f1-border rounded-lg p-6 mb-8 racing-stripe">
          <div className="pl-4 flex items-start justify-between flex-wrap gap-4">
            <div>
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-1">
                Recommended Strategy
              </div>
              <div className="font-display font-black text-3xl text-white">
                {cleanName(best.strategy_name)}
              </div>
              <div className="font-body text-f1-muted mt-1">
                {best.compound_sequence.replace(/→/g, ' → ')} · {best.num_stops}-stop
              </div>
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
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Delta bar chart */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h4 className="font-display font-bold text-lg mb-6">
              Strategy Rankings — Delta to Best
            </h4>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 90, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A3A" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: '#8A8A9A', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#2A2A3A' }}
                  tickLine={false}
                  label={{ value: 'Delta (seconds)', position: 'bottom', fill: '#8A8A9A', fontSize: 11 }}
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
              Confidence Intervals (P5–P95)
            </h4>
            <div className="space-y-6 mt-8">
              {strategies.slice(0, 5).map((s, i) => {
                const minP5 = Math.min(...strategies.slice(0, 5).map(x => x.p5_time))
                const maxP95 = Math.max(...strategies.slice(0, 5).map(x => x.p95_time))
                const range = maxP95 - minP5 || 1
                const leftPct = ((s.p5_time - minP5) / range) * 100
                const widthPct = ((s.p95_time - s.p5_time) / range) * 100
                const medianPct = ((s.median_time - minP5) / range) * 100

                return (
                  <div key={i}>
                    <div className="flex justify-between mb-1">
                      <span className="font-mono text-xs text-f1-muted">{cleanName(s.strategy_name)}</span>
                      <span className="font-mono text-xs text-f1-border">
                        {s.p5_time.toFixed(0)}s – {s.p95_time.toFixed(0)}s
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
                <span className="font-mono text-xs text-f1-muted">P5–P95 range</span>
              </div>
            </div>
          </div>
        </div>

        {/* Strategy table */}
        <div className="bg-f1-card border border-f1-border rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-f1-border">
                {['Rank', 'Strategy', 'Stops', 'Median', 'Delta', 'Std Dev', 'SC Events'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-mono text-xs text-f1-muted uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {strategies.map((s, i) => {
                const delta = Math.round((s.median_time - bestMedian) * 10) / 10
                return (
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
                      <div className="font-display font-bold text-sm">{cleanName(s.strategy_name)}</div>
                      <div className="font-mono text-xs text-f1-border">{s.compound_sequence}</div>
                    </td>
                    <td className="px-4 py-3 font-mono text-sm">{s.num_stops}</td>
                    <td className="px-4 py-3 font-mono text-sm">{s.median_time.toFixed(1)}s</td>
                    <td className="px-4 py-3">
                      <span className={`font-mono text-sm ${i === 0 ? 'text-green-400 font-bold' : 'text-f1-muted'}`}>
                        {i === 0 ? 'BEST' : `+${delta.toFixed(1)}s`}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-sm text-f1-muted">±{s.std_time.toFixed(1)}</td>
                    <td className="px-4 py-3 font-mono text-sm text-f1-muted">{s.mean_sc_events.toFixed(1)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
