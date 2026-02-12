'use client'

import { useState } from 'react'
import { rlResults, CircuitRLResult, RLSampleRace } from '@/data/rl'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend,
  ReferenceLine, Area, AreaChart,
} from 'recharts'

interface Props {
  circuitKey: string
}

// Compound colours
const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#ff3333',
  MEDIUM: '#ffd700',
  HARD: '#e0e0e0',
}

function CompoundPill({ compound }: { compound: string }) {
  const bg = COMPOUND_COLORS[compound] || '#888'
  const text = compound === 'HARD' ? '#111' : compound === 'MEDIUM' ? '#111' : '#fff'
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-[10px] font-mono font-bold leading-none"
      style={{ backgroundColor: bg, color: text }}
    >
      {compound[0]}
    </span>
  )
}

function StatCard({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: boolean
}) {
  return (
    <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
      <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
        {label}
      </div>
      <div className={`font-display font-black text-2xl ${accent ? 'text-f1-red' : 'text-white'}`}>
        {value}
      </div>
      {sub && (
        <div className="font-mono text-xs text-f1-muted mt-1">{sub}</div>
      )}
    </div>
  )
}

function WinRateBar({ rlWin, mcWin }: { rlWin: number; mcWin: number }) {
  return (
    <div>
      <div className="flex justify-between font-mono text-xs text-f1-muted mb-1">
        <span>RL Agent</span>
        <span>MC Optimizer</span>
      </div>
      <div className="h-6 rounded-full overflow-hidden flex bg-f1-darker border border-f1-border">
        <div
          className="h-full bg-emerald-500 transition-all duration-700 flex items-center justify-center"
          style={{ width: `${rlWin}%` }}
        >
          {rlWin >= 15 && (
            <span className="font-mono text-[11px] font-bold text-white">
              {rlWin.toFixed(1)}%
            </span>
          )}
        </div>
        <div
          className="h-full bg-f1-red transition-all duration-700 flex items-center justify-center"
          style={{ width: `${mcWin}%` }}
        >
          {mcWin >= 15 && (
            <span className="font-mono text-[11px] font-bold text-white">
              {mcWin.toFixed(1)}%
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function LapTrace({ race, totalLaps }: { race: RLSampleRace; totalLaps: number }) {
  // Build lap-by-lap data for chart
  const data = race.tyreAges.map((age, i) => ({
    lap: i + 1,
    tyreAge: age,
    compound: race.compounds[i],
    isSC: race.scLaps.includes(i + 1),
    isVSC: race.vscLaps.includes(i + 1),
    isPit: race.pitLaps.includes(i + 1),
  }))

  // Starting compound is compounds[0]
  const startCompound = race.compounds[0] || 'MEDIUM'
  const delta = race.mcTime - race.totalTime

  return (
    <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <div className="font-mono text-xs text-f1-muted">
            {race.category} — {race.stops} stop{race.stops !== 1 ? 's' : ''} — {race.totalTime.toFixed(1)}s
          </div>
          {race.mcTime > 0 && (
            <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
              race.rlWon
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}>
              {race.rlWon ? 'RL' : 'MC'} {race.rlWon ? '+' : ''}{delta.toFixed(1)}s
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <CompoundPill compound={startCompound} />
          {race.pitLaps.map((lap, i) => {
            const compAfter = race.compounds[lap] || race.compounds[lap - 1]
            return (
              <span key={i} className="flex items-center gap-1 font-mono text-[10px] text-f1-muted">
                → L{lap} → <CompoundPill compound={compAfter} />
              </span>
            )
          })}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={100}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="tyreGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="lap" tick={{ fontSize: 9, fill: '#666' }}
            tickLine={false} axisLine={false}
          />
          <YAxis tick={{ fontSize: 9, fill: '#666' }} tickLine={false} axisLine={false} />
          {/* SC zones */}
          {race.scLaps.map(lap => (
            <ReferenceLine key={`sc-${lap}`} x={lap} stroke="#ff3333" strokeDasharray="2 2" strokeOpacity={0.5} />
          ))}
          {/* Pit laps */}
          {race.pitLaps.map(lap => (
            <ReferenceLine key={`pit-${lap}`} x={lap} stroke="#ffd700" strokeWidth={2} strokeOpacity={0.8} />
          ))}
          <Area
            type="monotone" dataKey="tyreAge" stroke="#10b981"
            fill="url(#tyreGrad)" strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-1 font-mono text-[10px] text-f1-muted">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-emerald-500" /> Tyre age
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-yellow-400" /> Pit stop
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-f1-red opacity-50" style={{ borderBottom: '1px dashed' }} /> Safety Car
        </span>
      </div>
    </div>
  )
}

export function RLView({ circuitKey }: Props) {
  const [selectedRace, setSelectedRace] = useState(0)

  // Look up data with flexible key matching
  const data: CircuitRLResult | undefined =
    rlResults[circuitKey] ||
    rlResults[`${circuitKey}_2025`] ||
    Object.values(rlResults).find(r => r.circuitKey === circuitKey)

  if (!data) {
    return (
      <section id="rl" className="py-24 border-b border-f1-border">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-8">
            <div className="font-mono text-xs text-emerald-400 uppercase tracking-widest mb-3">
              Reinforcement Learning
            </div>
            <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
              RL Agent Comparison
            </h2>
          </div>
          <div className="bg-f1-card border border-f1-border rounded-lg p-12 text-center">
            <div className="font-mono text-f1-muted text-sm">
              No RL data available for this circuit yet.
              <br />
              <span className="text-f1-border text-xs mt-2 block">
                Train an agent: python -m src.rl.train --circuit {circuitKey}
              </span>
            </div>
          </div>
        </div>
      </section>
    )
  }

  const { mc, rl, comparison } = data
  // Use actual median times for the headline — not per-race delta which can mislead
  const medianDiff = mc.medianTime - rl.medianTime  // positive = RL faster
  const rlFasterOverall = medianDiff > 0
  const absDiff = Math.abs(medianDiff)

  // Stop distribution chart data
  const stopDist = rl.stopDistribution
    ? Object.entries(rl.stopDistribution).map(([stops, count]) => ({
        stops: `${stops}-stop`,
        count: Number(count),
        pct: Math.round((Number(count) / data.nRaces) * 100),
      }))
    : []

  // Time distribution comparison
  const timeComparison = [
    { metric: 'P5 (best)', mc: mc.p5Time, rl: rl.p5Time },
    { metric: 'Median', mc: mc.medianTime, rl: rl.medianTime },
    { metric: 'Mean', mc: mc.meanTime, rl: rl.meanTime },
    { metric: 'P95 (worst)', mc: mc.p95Time, rl: rl.p95Time },
  ]

  return (
    <section id="rl" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-emerald-400 uppercase tracking-widest mb-3">
            Reinforcement Learning
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            RL Agent vs Monte Carlo
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            A PPO agent trained entirely in simulation learns lap-by-lap pit stop timing.
            Compared head-to-head against the MC-optimized fixed strategy on{' '}
            <span className="text-white">{data.nRaces.toLocaleString()} identical race simulations</span>.
          </p>
        </div>

        {/* Headline result */}
        <div className={`bg-f1-card border rounded-lg p-6 mb-8 ${
          rlFasterOverall ? 'border-emerald-500/40' : 'border-f1-red/40'
        }`}>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="font-mono text-xs text-f1-muted uppercase tracking-widest mb-1">
                Headline Result (Median Race Time)
              </div>
              <div className="font-display font-black text-3xl">
                {rlFasterOverall ? (
                  <span className="text-emerald-400">
                    RL faster by {absDiff.toFixed(1)}s
                  </span>
                ) : absDiff < 1 ? (
                  <span className="text-yellow-400">Effectively tied ({absDiff.toFixed(1)}s)</span>
                ) : (
                  <span className="text-f1-red">
                    MC faster by {absDiff.toFixed(1)}s
                  </span>
                )}
              </div>
              <div className="font-mono text-xs text-f1-muted mt-1">
                RL wins {comparison.rlWinRate}% of {data.nRaces} head-to-head races
              </div>
            </div>
            <div className="w-full md:w-80">
              <WinRateBar rlWin={comparison.rlWinRate} mcWin={comparison.mcWinRate} />
            </div>
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <StatCard
            label="RL Median"
            value={`${rl.medianTime.toFixed(1)}s`}
            sub={`σ = ${rl.stdTime.toFixed(1)}s`}
            accent={rlFasterOverall}
          />
          <StatCard
            label="MC Median"
            value={`${mc.medianTime.toFixed(1)}s`}
            sub={`${data.mcStrategy} (${data.mcStops}-stop)`}
          />
          <StatCard
            label="SC Race Win Rate"
            value={`${comparison.scRaceRlWinRate}%`}
            sub={`${comparison.scRaces} races with SC`}
            accent={comparison.scRaceRlWinRate > 50}
          />
          <StatCard
            label="No-SC Win Rate"
            value={`${comparison.noScRaceRlWinRate}%`}
            sub={`${comparison.noScRaces} clean races`}
            accent={comparison.noScRaceRlWinRate > 50}
          />
        </div>

        {/* Two-column: time distribution + stop distribution */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Time distribution comparison */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h3 className="font-display font-bold text-lg mb-4">
              Race Time Distribution
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={timeComparison} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#888' }} />
                <YAxis
                  tick={{ fontSize: 10, fill: '#888' }}
                  domain={['auto', 'auto']}
                  tickFormatter={(v: number) => `${v.toFixed(0)}s`}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number, name: string) => [`${value.toFixed(1)}s`, name]}
                />
                <Bar dataKey="mc" fill="#e10600" name="MC" radius={[4, 4, 0, 0]} barSize={24} />
                <Bar dataKey="rl" fill="#10b981" name="RL" radius={[4, 4, 0, 0]} barSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* RL stop distribution */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h3 className="font-display font-bold text-lg mb-2">
              RL Stop Distribution
            </h3>
            <p className="font-mono text-xs text-f1-muted mb-4">
              Mean: {rl.meanStops?.toFixed(2)} stops — MC always uses {data.mcStops}
            </p>

            {stopDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={stopDist} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="stops" tick={{ fontSize: 11, fill: '#888' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#888' }} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                    formatter={(value: number) => [`${value} races`, 'Count']}
                  />
                  <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} barSize={40}>
                    {stopDist.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#10b981' : i === 1 ? '#3b82f6' : '#8b5cf6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[200px] flex items-center justify-center text-f1-muted font-mono text-sm">
                No distribution data
              </div>
            )}
          </div>
        </div>

        {/* Lap-by-lap traces */}
        {data.sampleRaces && data.sampleRaces.length > 0 && (
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-display font-bold text-lg">
                  Lap-by-Lap Agent Decisions
                </h3>
                <p className="font-mono text-xs text-f1-muted mt-1">
                  Tyre age trace with pit stops and safety car events
                </p>
              </div>
              <div className="flex gap-2 flex-wrap">
                {data.sampleRaces.map((race, i) => {
                  const label = race.category || `Race ${i + 1}`
                  const hasSC = race.scLaps.length > 0
                  return (
                    <button
                      key={i}
                      onClick={() => setSelectedRace(i)}
                      className={`px-3 h-8 rounded font-mono text-[11px] transition-all whitespace-nowrap ${
                        selectedRace === i
                          ? hasSC
                            ? 'bg-amber-500 text-black'
                            : 'bg-emerald-500 text-white'
                          : 'bg-f1-darker border border-f1-border text-f1-muted hover:text-white'
                      }`}
                    >
                      {label}
                    </button>
                  )
                })}
              </div>
            </div>

            <LapTrace
              race={data.sampleRaces[selectedRace]}
              totalLaps={data.sampleRaces[selectedRace].tyreAges.length}
            />
          </div>
        )}

        {/* Method note */}
        <div className="mt-6 font-mono text-xs text-f1-border leading-relaxed">
          PPO agent (2×128 MLP, γ=0.998) trained for 500K timesteps in the same simulation
          environment used by the Monte Carlo optimizer. Both approaches share identical
          physics: XGBoost degradation model, Bayesian SC priors, fuel correction.
          The RL agent makes lap-by-lap pit/stay decisions; MC uses pre-optimized fixed strategies.
        </div>
      </div>
    </section>
  )
}
