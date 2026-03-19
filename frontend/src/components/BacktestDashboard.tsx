'use client'

import { useState } from 'react'
import { backtestData, type BacktestRace } from '@/data/backtestResults'
import { useChartTheme } from '@/lib/ThemeProvider'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis,
} from 'recharts'

const VERDICT_CONFIG = {
  EXACT:      { label: 'Exact Match',   color: '#22c55e', icon: '★', desc: 'Compounds + stops correct' },
  PARTIAL:    { label: 'Partial Match',  color: '#3b82f6', icon: '◐', desc: 'Stops correct, similar compounds' },
  STOPS_ONLY: { label: 'Stops Only',     color: '#f59e0b', icon: '✓', desc: 'Stop count correct, different compounds' },
  MISS:       { label: 'Miss',           color: '#ef4444', icon: '✗', desc: 'Wrong stop count' },
  WET:        { label: 'Wet Race',       color: '#6b7280', icon: '☁', desc: 'Wet race — model not applicable' },
} as const

const COMPOUND_COLORS: Record<string, string> = {
  S: '#E10600', M: '#FFD700', H: '#a0a0ae', I: '#22c55e', W: '#3b82f6',
}

function CompoundPill({ compound }: { compound: string }) {
  const color = COMPOUND_COLORS[compound] || '#888'
  const isLight = compound === 'M' || compound === 'H'
  return (
    <span
      className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold font-mono"
      style={{ backgroundColor: color, color: isLight ? '#111' : '#fff' }}
    >
      {compound}
    </span>
  )
}

function CompoundSequence({ norm }: { norm: string }) {
  const parts = norm.split('-')
  return (
    <div className="flex items-center gap-1">
      {parts.map((c, i) => (
        <span key={i} className="flex items-center gap-0.5">
          <CompoundPill compound={c} />
          {i < parts.length - 1 && <span className="text-f1-muted text-xs">→</span>}
        </span>
      ))}
    </div>
  )
}

function VerdictBadge({ verdict }: { verdict: BacktestRace['verdict'] }) {
  const cfg = VERDICT_CONFIG[verdict]
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-bold"
      style={{ backgroundColor: `${cfg.color}20`, color: cfg.color, border: `1px solid ${cfg.color}40` }}
    >
      <span>{cfg.icon}</span>
      {cfg.label}
    </span>
  )
}

export function BacktestDashboard() {
  const chart = useChartTheme()
  const { overall, seasonSummaries, races } = backtestData
  const [selectedSeason, setSelectedSeason] = useState<number | 'all'>('all')

  const filteredRaces = selectedSeason === 'all'
    ? races
    : races.filter(r => r.season === selectedSeason)

  const dryRaces = filteredRaces.filter(r => !r.isWet)

  // Verdict distribution for chart
  const verdictData = Object.entries(VERDICT_CONFIG).map(([key, cfg]) => ({
    name: cfg.label,
    count: filteredRaces.filter(r => r.verdict === key).length,
    color: cfg.color,
  })).filter(d => d.count > 0)

  // Season comparison data
  const seasonCompData = seasonSummaries.map(s => ({
    season: `${s.trainSeasons.join('-')} → ${s.season}`,
    shortLabel: String(s.season),
    stopsMatch: s.stopsMatchRate,
    compoundExact: s.compoundExactRate,
    compoundPartial: s.compoundPartialRate,
    pitError: s.avgPitLapError,
  }))

  // Pit lap error scatter data
  const pitScatterData = dryRaces
    .filter(r => r.pitLapErrors.length > 0)
    .flatMap(r => r.pitLapErrors.map(e => ({
      predicted: e.predicted,
      actual: e.actual,
      error: e.error,
      circuit: r.circuit.replace(' Grand Prix', ''),
      season: r.season,
    })))

  return (
    <section id="backtest" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Backtesting
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Predicted vs. Actual
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            Enriched validation comparing our model&apos;s top strategy against what the race winner
            actually used — beyond just stop count, we check compound sequences and pit lap timing.
          </p>
        </div>

        {/* Overall stats cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-10">
          {[
            { value: `${overall.stopsMatchRate}%`, label: 'Stops Match', sub: `${overall.dryRaces} dry races` },
            { value: `${overall.compoundExactRate}%`, label: 'Compound Exact', sub: 'Same compounds + order' },
            { value: `${overall.compoundPartialRate}%`, label: 'Compound Partial', sub: 'Same compounds used' },
            { value: overall.avgPitLapError !== null ? `${overall.avgPitLapError}` : 'N/A', label: 'Avg Pit Error', sub: 'Laps off target' },
            { value: `${overall.wetRaces}`, label: 'Wet Excluded', sub: 'Model not applicable' },
          ].map((stat, i) => (
            <div key={i} className="theme-card rounded-xl p-4 text-center">
              <div className="font-display font-black text-3xl text-f1-light">{stat.value}</div>
              <div className="font-mono text-xs text-f1-red uppercase tracking-wider mt-1">{stat.label}</div>
              <div className="font-body text-xs text-f1-muted mt-1">{stat.sub}</div>
            </div>
          ))}
        </div>

        {/* Season filter */}
        <div className="flex gap-2 mb-8">
          <button
            onClick={() => setSelectedSeason('all')}
            className={`px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
              selectedSeason === 'all'
                ? 'bg-f1-red text-white'
                : 'theme-card text-f1-muted hover:text-f1-light'
            }`}
          >
            All Seasons
          </button>
          {[2023, 2024, 2025].map(s => (
            <button
              key={s}
              onClick={() => setSelectedSeason(s)}
              className={`px-4 py-2 rounded-lg font-mono text-xs transition-colors ${
                selectedSeason === s
                  ? 'bg-f1-red text-white'
                  : 'theme-card text-f1-muted hover:text-f1-light'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Charts row */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Verdict distribution */}
          <div className="theme-card rounded-xl p-6">
            <h4 className="font-display font-bold text-lg mb-6">Match Quality Distribution</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={verdictData} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                <XAxis
                  dataKey="name"
                  tick={{ fill: chart.text, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: chart.grid }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
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
                  formatter={(value: number) => [`${value} races`, 'Count']}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {verdictData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-3 mt-4 justify-center">
              {Object.entries(VERDICT_CONFIG).map(([key, cfg]) => (
                <div key={key} className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: cfg.color }} />
                  <span className="font-mono text-[10px] text-f1-muted">{cfg.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Season progression */}
          <div className="theme-card rounded-xl p-6">
            <h4 className="font-display font-bold text-lg mb-6">Accuracy by Season</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={seasonCompData} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                <XAxis
                  dataKey="shortLabel"
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: chart.grid }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, 100]}
                  tickFormatter={(v: number) => `${v}%`}
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
                  formatter={(value: number, name: string) => {
                    const labels: Record<string, string> = {
                      stopsMatch: 'Stops Match',
                      compoundExact: 'Compound Exact',
                      compoundPartial: 'Compound Partial',
                    }
                    return [`${value}%`, labels[name] || name]
                  }}
                />
                <Bar dataKey="stopsMatch" fill="#f59e0b" radius={[4, 4, 0, 0]} name="stopsMatch" />
                <Bar dataKey="compoundExact" fill="#22c55e" radius={[4, 4, 0, 0]} name="compoundExact" />
                <Bar dataKey="compoundPartial" fill="#3b82f6" radius={[4, 4, 0, 0]} name="compoundPartial" />
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-6 mt-4 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-amber-500" />
                <span className="font-mono text-xs text-f1-muted">Stops Match</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-green-500" />
                <span className="font-mono text-xs text-f1-muted">Compound Exact</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-blue-500" />
                <span className="font-mono text-xs text-f1-muted">Compound Partial</span>
              </div>
            </div>
          </div>
        </div>

        {/* Pit lap accuracy scatter */}
        {pitScatterData.length > 0 && (
          <div className="theme-card rounded-xl p-6 mb-12">
            <h4 className="font-display font-bold text-lg mb-2">Pit Lap Accuracy</h4>
            <p className="font-body text-sm text-f1-muted mb-6">
              Each dot is a predicted vs actual pit stop lap. Points on the diagonal = perfect prediction.
            </p>
            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart margin={{ left: 20, right: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chart.grid} />
                <XAxis
                  type="number"
                  dataKey="predicted"
                  name="Predicted Lap"
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={{ stroke: chart.grid }}
                  label={{ value: 'Predicted Pit Lap', position: 'bottom', fill: chart.text, fontSize: 12, fontFamily: 'var(--font-mono)' }}
                />
                <YAxis
                  type="number"
                  dataKey="actual"
                  name="Actual Lap"
                  tick={{ fill: chart.text, fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  axisLine={false}
                  label={{ value: 'Actual Pit Lap', angle: -90, position: 'insideLeft', fill: chart.text, fontSize: 12, fontFamily: 'var(--font-mono)' }}
                />
                <ZAxis range={[40, 40]} />
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
                  formatter={(value: number, name: string) => [
                    `Lap ${value}`,
                    name === 'predicted' ? 'Predicted' : name === 'actual' ? 'Actual' : name,
                  ]}
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null
                    const d = payload[0].payload
                    return (
                      <div style={{
                        background: chart.tooltipBg,
                        border: `1px solid ${chart.tooltipBorder}`,
                        borderRadius: '4px',
                        padding: '8px 12px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '11px',
                        color: chart.tooltipText,
                      }}>
                        <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.circuit} ({d.season})</div>
                        <div>Predicted: Lap {d.predicted}</div>
                        <div>Actual: Lap {d.actual}</div>
                        <div style={{ color: d.error <= 3 ? '#22c55e' : d.error <= 7 ? '#f59e0b' : '#ef4444' }}>
                          Error: {d.error} laps
                        </div>
                      </div>
                    )
                  }}
                />
                <Scatter data={pitScatterData} fill={chart.red} fillOpacity={0.7} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="text-center mt-2">
              <span className="font-mono text-xs text-f1-muted">
                Average pit lap error: {overall.avgPitLapError} laps across {pitScatterData.length} pit stops
              </span>
            </div>
          </div>
        )}

        {/* Race-by-race table */}
        <div className="theme-card rounded-xl p-6">
          <h4 className="font-display font-bold text-lg mb-6">
            Race-by-Race Breakdown
            <span className="font-mono text-sm text-f1-muted ml-3">
              ({filteredRaces.length} races)
            </span>
          </h4>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-f1-border">
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Season</th>
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Circuit</th>
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Winner</th>
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Actual Strategy</th>
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Predicted Strategy</th>
                  <th className="text-left py-3 px-2 font-mono text-xs text-f1-muted uppercase">Pit Laps (A→P)</th>
                  <th className="text-center py-3 px-2 font-mono text-xs text-f1-muted uppercase">Pit Error</th>
                  <th className="text-center py-3 px-2 font-mono text-xs text-f1-muted uppercase">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {filteredRaces.map((race, i) => (
                  <tr key={i} className="border-b border-f1-border/50 hover:bg-f1-dark/30 transition-colors">
                    <td className="py-3 px-2 font-mono text-xs text-f1-muted">{race.season}</td>
                    <td className="py-3 px-2 font-body text-sm text-f1-light">
                      {race.circuit.replace(' Grand Prix', '')}
                    </td>
                    <td className="py-3 px-2 font-mono text-xs font-bold text-f1-light">{race.winner}</td>
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-f1-muted">{race.actualStops}s</span>
                        <CompoundSequence norm={race.actualCompoundNorm} />
                      </div>
                    </td>
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-f1-muted">{race.predictedStops}s</span>
                        <CompoundSequence norm={race.predictedCompoundNorm} />
                      </div>
                    </td>
                    <td className="py-3 px-2 font-mono text-xs text-f1-muted">
                      {race.actualPitLaps.length > 0 && race.predictedPitLaps.length > 0
                        ? `${race.actualPitLaps.join(',')} → ${race.predictedPitLaps.join(',')}`
                        : '—'
                      }
                    </td>
                    <td className="py-3 px-2 text-center">
                      {race.pitLapMeanError !== null ? (
                        <span
                          className="font-mono text-xs font-bold"
                          style={{
                            color: race.pitLapMeanError <= 3 ? '#22c55e'
                              : race.pitLapMeanError <= 7 ? '#f59e0b'
                              : '#ef4444'
                          }}
                        >
                          {race.pitLapMeanError}
                        </span>
                      ) : (
                        <span className="font-mono text-xs text-f1-muted">—</span>
                      )}
                    </td>
                    <td className="py-3 px-2 text-center">
                      <VerdictBadge verdict={race.verdict} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Methodology note */}
        <div className="mt-8 p-6 rounded-xl border border-f1-border/50">
          <h4 className="font-display font-bold text-sm mb-3 text-f1-muted uppercase tracking-wider">
            Methodology
          </h4>
          <div className="font-body text-sm text-f1-muted leading-relaxed space-y-2">
            <p>
              Each fold trains a fresh XGBoost degradation model on historical seasons only, then runs
              Monte Carlo strategy optimization for every circuit in the validation season. The model&apos;s
              top-ranked strategy is compared against what the actual race winner used.
            </p>
            <p>
              <strong className="text-f1-light">Verdict levels:</strong>{' '}
              <span style={{ color: '#22c55e' }}>Exact</span> = same compounds in same order;{' '}
              <span style={{ color: '#3b82f6' }}>Partial</span> = correct stops + matching compound set;{' '}
              <span style={{ color: '#f59e0b' }}>Stops Only</span> = stop count matches;{' '}
              <span style={{ color: '#ef4444' }}>Miss</span> = wrong stop count.
              Wet races are excluded since the model only handles dry compounds.
            </p>
            <p>
              <strong className="text-f1-light">Pit lap error</strong> measures how far off our predicted
              pit stop laps are from the winner&apos;s actual pit laps (matched greedily by closest lap).
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
