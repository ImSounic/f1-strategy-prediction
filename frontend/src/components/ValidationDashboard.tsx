'use client'

import { validationData } from '@/data/strategies'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend,
} from 'recharts'

export function ValidationDashboard() {
  const { folds, models, shapFeatures } = validationData

  // Learning curve data
  const learningData = folds.map(f => ({
    label: f.label,
    stints: f.trainStints,
    exactMatch: f.exactMatch,
    top5Match: f.top5Match,
    mae: f.cvMae,
  }))

  return (
    <section id="validation" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Model Validation
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Zero Data Leakage
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-xl">
            Rolling temporal validation. Each fold trains only on past data
            and predicts on a completely unseen future season.
          </p>
        </div>

        {/* Validation folds */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {folds.map((fold, i) => (
            <div
              key={i}
              className={`bg-f1-card border rounded-lg p-6 ${
                i === 2 ? 'border-f1-red glow-red' : 'border-f1-border'
              }`}
            >
              <div className="font-mono text-xs text-f1-muted uppercase tracking-wider mb-4">
                {fold.label}
              </div>

              <div className="flex items-baseline gap-1 mb-1">
                <span className="font-display font-black text-4xl text-white">
                  {fold.exactMatch}%
                </span>
                <span className="font-body text-f1-muted">exact</span>
              </div>

              <div className="flex items-baseline gap-1 mb-4">
                <span className="font-display font-bold text-2xl text-blue-400">
                  {fold.top5Match}%
                </span>
                <span className="font-body text-f1-muted text-sm">top-5</span>
              </div>

              <div className="pt-4 border-t border-f1-border space-y-2">
                <div className="flex justify-between">
                  <span className="font-mono text-xs text-f1-muted">Training stints</span>
                  <span className="font-mono text-xs text-white">{new Intl.NumberFormat('en-US').format(fold.trainStints)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-mono text-xs text-f1-muted">CV MAE</span>
                  <span className="font-mono text-xs text-white">{fold.cvMae.toFixed(4)}s</span>
                </div>
              </div>

              {i === 2 && (
                <div className="mt-4 pt-4 border-t border-f1-red/30">
                  <div className="font-mono text-xs text-f1-red uppercase tracking-wider">
                    ★ Headline result — completely unseen 2025
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Charts row */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Learning curve */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h4 className="font-display font-bold text-lg mb-6">
              Accuracy Improves with Data
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={learningData} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A3A" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#8A8A9A', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#2A2A3A' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#8A8A9A', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, 100]}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1A1A26',
                    border: '1px solid #2A2A3A',
                    borderRadius: '4px',
                    fontFamily: 'JetBrains Mono',
                    fontSize: '12px',
                  }}
                  formatter={(value: number, name: string) => [
                    `${value}%`,
                    name === 'exactMatch' ? 'Exact Match' : 'Top 5 Match',
                  ]}
                />
                <Bar dataKey="exactMatch" fill="#E10600" radius={[4, 4, 0, 0]} name="exactMatch" />
                <Bar dataKey="top5Match" fill="#0090D0" radius={[4, 4, 0, 0]} name="top5Match" />
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-6 mt-4 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-[#E10600]" />
                <span className="font-mono text-xs text-f1-muted">Exact Match</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-[#0090D0]" />
                <span className="font-mono text-xs text-f1-muted">Top 5 Match</span>
              </div>
            </div>
          </div>

          {/* Model comparison */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h4 className="font-display font-bold text-lg mb-6">
              Model Comparison — MAE
            </h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={models} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A3A" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#8A8A9A', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#2A2A3A' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#8A8A9A', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0.06, 0.10]}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1A1A26',
                    border: '1px solid #2A2A3A',
                    borderRadius: '4px',
                    fontFamily: 'JetBrains Mono',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`${value.toFixed(4)}s`, 'MAE']}
                />
                <Bar dataKey="mae" radius={[4, 4, 0, 0]}>
                  {models.map((_, i) => (
                    <Cell
                      key={i}
                      fill={['#6B5B95', '#E10600', '#FF8700'][i]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="text-center mt-4">
              <span className="font-mono text-xs text-f1-muted">
                XGBoost wins: lowest MAE (0.0755s), fast training (3.2s)
              </span>
            </div>
          </div>
        </div>

        {/* SHAP feature importance */}
        <div className="bg-f1-card border border-f1-border rounded-lg p-6">
          <h4 className="font-display font-bold text-lg mb-2">
            SHAP Feature Importance
          </h4>
          <p className="font-body text-sm text-f1-muted mb-6">
            What drives tyre degradation? SHAP values reveal feature-level impact on model predictions.
          </p>
          <div className="space-y-3">
            {shapFeatures.map((feat, i) => {
              const maxVal = shapFeatures[0].importance
              const pct = (feat.importance / maxVal) * 100

              return (
                <div key={i} className="flex items-center gap-4">
                  <div className="w-32 font-mono text-xs text-f1-muted text-right">
                    {feat.name}
                  </div>
                  <div className="flex-1 h-5 bg-f1-darker rounded overflow-hidden">
                    <div
                      className="h-full rounded transition-all duration-700"
                      style={{
                        width: `${pct}%`,
                        background: i < 3
                          ? '#E10600'
                          : i < 6
                          ? 'linear-gradient(90deg, #E10600, #FF8700)'
                          : '#0090D0',
                      }}
                    />
                  </div>
                  <div className="w-16 font-mono text-xs text-white text-right">
                    {feat.importance.toFixed(4)}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-6 pt-4 border-t border-f1-border">
            <p className="font-body text-xs text-f1-border leading-relaxed max-w-2xl">
              Weather conditions (humidity, wind, temperature) dominate — confirming that environmental
              factors affect tyre degradation more than circuit layout. The manually collected
              asphalt_grip from Pirelli infographics ranks 3rd, validating the manual data collection effort.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
