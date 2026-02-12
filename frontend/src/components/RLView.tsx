'use client'

import { useState } from 'react'
import { rlResults, CircuitRLResult, RLSampleRace } from '@/data/rl'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend,
  ReferenceLine, Area, AreaChart, ComposedChart,
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

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

function formatDelta(seconds: number): string {
  const sign = seconds >= 0 ? '+' : '-'
  const abs = Math.abs(seconds)
  if (abs >= 60) {
    const m = Math.floor(abs / 60)
    const s = (abs % 60).toFixed(1)
    return `${sign}${m}m${parseFloat(s).toFixed(1)}s`
  }
  return `${sign}${abs.toFixed(1)}s`
}

function StrategyPills({ startCompound, pitLaps, compounds, label, color }: {
  startCompound: string; pitLaps: number[]; compounds: string[]; label: string; color: string
}) {
  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className={`font-mono text-[10px] font-bold mr-1`} style={{ color }}>{label}</span>
      <CompoundPill compound={startCompound} />
      {pitLaps.map((lap, i) => {
        const compAfter = compounds[lap] || compounds[lap - 1] || compounds[compounds.length - 1]
        return (
          <span key={i} className="flex items-center gap-1 font-mono text-[10px] text-f1-muted">
            → L{lap} → <CompoundPill compound={compAfter} />
          </span>
        )
      })}
    </div>
  )
}

function LapTrace({ race, totalLaps, mcStrategy }: { race: RLSampleRace; totalLaps: number; mcStrategy: string }) {
  // Build lap-by-lap data with both RL and MC tyre ages
  const hasMcData = race.mcTyreAges && race.mcTyreAges.length > 0
  const data = race.tyreAges.map((age, i) => ({
    lap: i + 1,
    rlTyreAge: age,
    mcTyreAge: hasMcData ? (race.mcTyreAges[i] ?? 0) : undefined,
    isSC: race.scLaps.includes(i + 1),
    isPit: race.pitLaps.includes(i + 1),
  }))

  const startCompound = race.compounds[0] || 'MEDIUM'
  const mcStartCompound = hasMcData ? (race.mcCompounds[0] || 'MEDIUM') : 'MEDIUM'
  const delta = race.mcTime - race.totalTime

  return (
    <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
      {/* Header: category, times, delta */}
      <div className="flex items-center justify-between mb-1 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <div className="font-mono text-xs text-f1-muted">
            {race.category} — {race.stops} stop{race.stops !== 1 ? 's' : ''}
          </div>
          {race.mcTime > 0 && (
            <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
              race.rlWon
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}>
              {race.rlWon ? 'RL wins' : 'MC wins'} by {formatDelta(Math.abs(delta)).replace(/^[+-]/, '')}
            </span>
          )}
        </div>
      </div>

      {/* Strategy comparison: RL vs MC */}
      <div className="flex flex-col gap-1 mb-3">
        <div className="flex items-center justify-between">
          <StrategyPills
            startCompound={startCompound}
            pitLaps={race.pitLaps}
            compounds={race.compounds}
            label="RL"
            color="#10b981"
          />
          <span className="font-mono text-[11px] text-emerald-400">{formatTime(race.totalTime)}</span>
        </div>
        {hasMcData && (
          <div className="flex items-center justify-between">
            <StrategyPills
              startCompound={mcStartCompound}
              pitLaps={race.mcPitLaps}
              compounds={race.mcCompounds}
              label="MC"
              color="#f59e0b"
            />
            <span className="font-mono text-[11px] text-amber-400">{formatTime(race.mcTime)}</span>
          </div>
        )}
      </div>

      {/* Chart with both lines */}
      <ResponsiveContainer width="100%" height={120}>
        <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="rlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
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
          {/* RL pit laps */}
          {race.pitLaps.map(lap => (
            <ReferenceLine key={`pit-${lap}`} x={lap} stroke="#10b981" strokeWidth={2} strokeOpacity={0.6} />
          ))}
          {/* MC pit laps */}
          {hasMcData && race.mcPitLaps.map(lap => (
            <ReferenceLine key={`mc-pit-${lap}`} x={lap} stroke="#f59e0b" strokeWidth={2} strokeOpacity={0.6} strokeDasharray="4 2" />
          ))}
          {/* MC tyre age line */}
          {hasMcData && (
            <Line
              type="monotone" dataKey="mcTyreAge" stroke="#f59e0b"
              strokeWidth={1.5} strokeOpacity={0.7} strokeDasharray="4 2"
              dot={false} isAnimationActive={false}
            />
          )}
          {/* RL tyre age area */}
          <Area
            type="monotone" dataKey="rlTyreAge" stroke="#10b981"
            fill="url(#rlGrad)" strokeWidth={1.5}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-1 font-mono text-[10px] text-f1-muted">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-emerald-500" /> RL tyre age
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-amber-400 opacity-70" style={{ borderBottom: '1px dashed' }} /> MC tyre age
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
                    RL faster by {formatDelta(absDiff)}
                  </span>
                ) : absDiff < 1 ? (
                  <span className="text-yellow-400">Effectively tied ({formatDelta(absDiff)})</span>
                ) : (
                  <span className="text-f1-red">
                    MC faster by {formatDelta(absDiff)}
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
            value={formatTime(rl.medianTime)}
            sub={`σ = ${rl.stdTime.toFixed(1)}s`}
            accent={rlFasterOverall}
          />
          <StatCard
            label="MC Median"
            value={formatTime(mc.medianTime)}
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
                  tickFormatter={(v: number) => formatTime(v)}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number, name: string) => [formatTime(value), name]}
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
              mcStrategy={data.mcStrategy}
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