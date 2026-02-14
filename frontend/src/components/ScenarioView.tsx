'use client'

import { useState } from 'react'
import { scenarioData } from '@/data/scenarios'
import type { ScenarioResult, DecisionTrigger, ScenarioStrategy } from '@/data/scenarios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

interface Props {
  circuitKey: string
}

function CompoundPill({ compound }: { compound: string }) {
  const bgMap: Record<string, string> = {
    S: 'bg-red-500 text-white',
    M: 'bg-yellow-400 text-gray-900',
    H: 'bg-white text-gray-900',
  }
  const cls = bgMap[compound] || 'bg-f1-border text-white'
  return (
    <span className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[9px] font-mono font-bold ${cls}`}>
      {compound}
    </span>
  )
}

function CompoundSequence({ compounds }: { compounds: string }) {
  const parts = compounds.replace(/\s/g, '').split('→')
  return (
    <div className="flex items-center gap-0.5">
      {parts.map((comp, i) => (
        <div key={i} className="flex items-center gap-0.5">
          <CompoundPill compound={comp} />
          {i < parts.length - 1 && (
            <span className="text-f1-border text-[10px]">→</span>
          )}
        </div>
      ))}
    </div>
  )
}

function TriggerCard({ trigger }: { trigger: DecisionTrigger }) {
  const impactColor = trigger.timeSaved >= 10
    ? 'border-red-500/40 bg-red-500/5'
    : trigger.timeSaved >= 5
      ? 'border-orange-500/40 bg-orange-500/5'
      : 'border-yellow-500/40 bg-yellow-500/5'

  return (
    <div className={`rounded-lg border p-4 ${impactColor}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{trigger.icon}</span>
            <span className="font-mono text-xs text-f1-muted uppercase tracking-wider">
              {(trigger.probability * 100).toFixed(0)}% chance
            </span>
          </div>
          <p className="font-body text-sm text-f1-light leading-snug">
            {trigger.trigger}
          </p>
        </div>
        <div className="text-right flex-shrink-0">
          <div className="font-display font-black text-xl text-green-400">
            +{trigger.timeSaved}s
          </div>
          <div className="font-mono text-[10px] text-f1-muted">
            saved vs default
          </div>
        </div>
      </div>
    </div>
  )
}

function ScenarioCard({
  scenario,
  isSelected,
  onClick,
  defaultPlanName,
}: {
  scenario: ScenarioResult
  isSelected: boolean
  onClick: () => void
  defaultPlanName: string
}) {
  const isDefault = scenario.defaultPlanRank === 1
  const needsSwitch = !isDefault

  return (
    <button
      onClick={onClick}
      className={`text-left rounded-lg border p-4 transition-all w-full ${
        isSelected
          ? 'border-f1-red bg-f1-red/10'
          : 'border-f1-border bg-f1-card hover:border-f1-muted'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xl">{scenario.icon}</span>
          <div>
            <div className="font-display font-bold text-sm text-white leading-tight">
              {scenario.scenarioName.split('(')[0].trim()}
            </div>
            <div className="font-mono text-[10px] text-f1-muted mt-0.5">
              {(scenario.probability * 100).toFixed(0)}% probability
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <div className="font-mono text-xs">
          {needsSwitch ? (
            <span className="text-orange-400">⚡ {scenario.scenarioBest}</span>
          ) : (
            <span className="text-green-400">✓ Default holds</span>
          )}
        </div>
        {needsSwitch && scenario.timeDelta > 0 && (
          <span className="font-mono text-xs text-green-400 font-bold">
            +{scenario.timeDelta}s
          </span>
        )}
      </div>
    </button>
  )
}

export function ScenarioView({ circuitKey }: Props) {
  const matchingKey = Object.keys(scenarioData)
    .find(k => k.startsWith(circuitKey + '_')) || ''

  const data = matchingKey ? scenarioData[matchingKey] : null

  const [selectedScenario, setSelectedScenario] = useState(0)

  if (!data) {
    return (
      <section id="scenarios" className="py-24 border-b border-f1-border">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-12">
            <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
              Contingency Planning
            </div>
            <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
              Race Scenarios
            </h2>
          </div>
          <div className="bg-f1-card border border-f1-border rounded-lg p-12 text-center">
            <div className="text-4xl mb-4">📋</div>
            <div className="font-display text-xl text-f1-muted mb-2">
              No scenario analysis for this circuit
            </div>
            <p className="font-body text-sm text-f1-border max-w-md mx-auto">
              Run <code className="text-f1-red">python -m src.scripts.precompute_scenarios --circuit {circuitKey}</code> to
              generate contingency plans.
            </p>
          </div>
        </div>
      </section>
    )
  }

  const scenario = data.scenarios[selectedScenario]
  const triggers = data.triggers

  // Chart data for selected scenario
  const chartData = scenario.strategies.slice(0, 6).map((s: ScenarioStrategy) => ({
    name: s.cleanName,
    delta: s.delta,
    stops: s.stops,
  }))

  return (
    <section id="scenarios" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Contingency Planning
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Race Scenarios
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            What if the plan meets reality? Each scenario shows the optimal strategy
            response. {data.nSims} simulations per scenario × {data.scenarios.length} conditions
            analysed.
          </p>
        </div>

        {/* Decision Triggers — the key actionable output */}
        {triggers.length > 0 && (
          <div className="mb-12">
            <h3 className="font-display font-bold text-xl mb-4 flex items-center gap-2">
              <span className="text-f1-red">⚡</span>
              Decision Triggers
            </h3>
            <p className="font-body text-sm text-f1-muted mb-4">
              When these scenarios occur, switching from the default plan ({data.defaultPlan.cleanName}) saves time.
              Sorted by impact.
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              {triggers.slice(0, 6).map((t: DecisionTrigger, i: number) => (
                <TriggerCard key={i} trigger={t} />
              ))}
            </div>
          </div>
        )}

        {/* Default plan banner */}
        <div className="bg-f1-card border border-f1-border rounded-lg p-5 mb-8 racing-stripe">
          <div className="pl-4 flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-1">
                Default Pre-Race Plan
              </div>
              <div className="flex items-center gap-3">
                <span className="font-display font-bold text-xl text-white">
                  {data.defaultPlan.cleanName}
                </span>
                <CompoundSequence compounds={data.defaultPlan.compounds} />
              </div>
            </div>
            <div className="text-right">
              <div className="font-mono text-xs text-f1-muted">
                SC probability at this circuit
              </div>
              <div className="font-display font-bold text-2xl text-white">
                {(data.scProbability * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Scenario grid + detail */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left: Scenario selector */}
          <div className="space-y-2">
            <h4 className="font-display font-bold text-sm uppercase tracking-wider text-f1-muted mb-3">
              Scenarios ({data.scenarios.length})
            </h4>
            {data.scenarios.map((sc: ScenarioResult, i: number) => (
              <ScenarioCard
                key={sc.scenarioId}
                scenario={sc}
                isSelected={i === selectedScenario}
                onClick={() => setSelectedScenario(i)}
                defaultPlanName={data.defaultPlan.cleanName}
              />
            ))}
          </div>

          {/* Right: Scenario detail */}
          <div className="lg:col-span-2 space-y-6">
            {/* Scenario header */}
            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <div className="flex items-start gap-3 mb-4">
                <span className="text-3xl">{scenario.icon}</span>
                <div>
                  <h4 className="font-display font-bold text-xl text-white">
                    {scenario.scenarioName}
                  </h4>
                  <p className="font-body text-sm text-f1-muted mt-1 leading-relaxed">
                    {scenario.description}
                  </p>
                </div>
              </div>

              {/* Key metrics for this scenario */}
              <div className="grid grid-cols-3 gap-4 pt-4 border-t border-f1-border">
                <div>
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">
                    Best Strategy
                  </div>
                  <div className="font-display font-bold text-lg text-white mt-1">
                    {scenario.scenarioBest}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">
                    Default Plan Rank
                  </div>
                  <div className={`font-display font-bold text-lg mt-1 ${
                    scenario.defaultPlanRank === 1 ? 'text-green-400' : 'text-orange-400'
                  }`}>
                    P{scenario.defaultPlanRank}
                    {scenario.defaultPlanRank > 1 && (
                      <span className="text-sm font-normal text-f1-muted ml-1">
                        (+{scenario.timeDelta}s)
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">
                    Probability
                  </div>
                  <div className="font-display font-bold text-lg text-white mt-1">
                    {(scenario.probability * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Strategy ranking chart */}
            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <h4 className="font-display font-bold text-lg mb-6">
                Strategy Ranking — {scenario.scenarioName.split('(')[0].trim()}
              </h4>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
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
                    width={75}
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
                    {chartData.map((entry: any, i: number) => (
                      <Cell
                        key={i}
                        fill={entry.stops === 1 ? '#E10600' : '#0090D0'}
                        opacity={i === 0 ? 1 : 0.65}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-6 mt-3 justify-center">
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

            {/* Strategy table for this scenario */}
            <div className="bg-f1-card border border-f1-border rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-f1-border">
                    {['Rank', 'Strategy', 'Compounds', 'Stops', 'Median', 'Delta', 'SC Events'].map(h => (
                      <th key={h} className="px-3 py-2 text-left font-mono text-[10px] text-f1-muted uppercase tracking-wider">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {scenario.strategies.map((s: ScenarioStrategy, i: number) => {
                    const isDefault = s.name === data.defaultPlan.name
                    return (
                      <tr
                        key={i}
                        className={`border-b border-f1-border/50 transition-colors hover:bg-f1-darker/50 ${
                          i === 0 ? 'bg-f1-red/5' : isDefault ? 'bg-blue-500/5' : ''
                        }`}
                      >
                        <td className="px-3 py-2">
                          <span className={`font-display font-bold text-sm ${
                            i === 0 ? 'text-f1-red' : 'text-f1-muted'
                          }`}>
                            P{s.rank}
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className="font-display font-bold text-xs">{s.cleanName}</span>
                            {isDefault && (
                              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                                DEFAULT
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-3 py-2">
                          <CompoundSequence compounds={s.compounds} />
                        </td>
                        <td className="px-3 py-2 font-mono text-xs">{s.stops}</td>
                        <td className="px-3 py-2 font-mono text-xs">{s.medianTime.toFixed(1)}s</td>
                        <td className="px-3 py-2">
                          <span className={`font-mono text-xs ${
                            i === 0 ? 'text-green-400 font-bold' : 'text-f1-muted'
                          }`}>
                            {i === 0 ? 'BEST' : `+${s.delta.toFixed(1)}s`}
                          </span>
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-f1-muted">
                          {s.scEvents.toFixed(1)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
