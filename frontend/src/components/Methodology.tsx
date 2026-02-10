'use client'

import { useState } from 'react'

interface Step {
  id: number
  phase: string
  title: string
  description: string
  details: string[]
  metrics: { label: string; value: string }[]
  color: string
  icon: string
}

const steps: Step[] = [
  {
    id: 1,
    phase: 'Phase 1',
    title: 'Data Ingestion',
    description: 'Collect comprehensive F1 race data from four complementary APIs, each providing unique telemetry and event information across 92 Grand Prix weekends.',
    details: [
      'FastF1 — lap-by-lap times, weather telemetry, track status flags',
      'OpenF1 — real-time stint data, pit stop durations',
      'Jolpica/Ergast — race results, qualifying, championship standings',
      'Pirelli — manually collected circuit characteristics from official infographics',
    ],
    metrics: [
      { label: 'Races', value: '92' },
      { label: 'Seasons', value: '2022–2025' },
      { label: 'APIs', value: '4' },
      { label: 'Stints', value: '4,250+' },
    ],
    color: '#0090D0',
    icon: '⟡',
  },
  {
    id: 2,
    phase: 'Phase 2',
    title: 'Feature Engineering',
    description: 'Transform raw lap times into meaningful degradation features using signal processing. The key innovation: Savitzky-Golay filtering extracts smooth degradation curves from noisy lap data.',
    details: [
      'Savitzky-Golay polynomial filtering to extract tyre degradation slopes',
      'Fuel-corrected lap times (removing ~0.055s/lap fuel burn effect)',
      'Safety car event detection and lap-level binary flags',
      'Circuit characteristic encoding from Pirelli data (8 features per track)',
    ],
    metrics: [
      { label: 'Features', value: '22' },
      { label: 'Target', value: 'DegSlope' },
      { label: 'Filter', value: 'Savitzky-Golay' },
      { label: 'Window', value: '7 laps' },
    ],
    color: '#00D2BE',
    icon: '◈',
  },
  {
    id: 3,
    phase: 'Phase 3',
    title: 'Model Training',
    description: 'Train and compare three regression models to predict tyre degradation rates. XGBoost wins on both accuracy and speed, confirmed by SHAP interpretability analysis.',
    details: [
      'XGBoost — gradient-boosted trees with RandomizedSearchCV (60 iterations)',
      'Ridge Regression — regularized linear baseline for comparison',
      'MLP Neural Network — 2-layer feedforward for non-linear comparison',
      'SHAP TreeExplainer for feature importance and model interpretability',
    ],
    metrics: [
      { label: 'Best MAE', value: '0.079s' },
      { label: 'Model', value: 'XGBoost' },
      { label: 'Train Time', value: '3.2s' },
      { label: 'HP Search', value: '60 iter' },
    ],
    color: '#E10600',
    icon: '⬢',
  },
  {
    id: 4,
    phase: 'Phase 4',
    title: 'Monte Carlo Simulation',
    description: 'Simulate thousands of race scenarios per strategy using stochastic safety car injection and Bayesian probability models. Each simulation draws random SC/VSC events calibrated to circuit-specific historical rates.',
    details: [
      'Bayesian Beta-Binomial model for per-circuit safety car probabilities',
      'Per-lap Bernoulli draws for stochastic SC/VSC event injection',
      'Strategy candidate generation: 1-stop and 2-stop with all compound permutations',
      'Fuel-adjusted and degradation-adjusted lap time computation',
    ],
    metrics: [
      { label: 'Speed', value: '9K/sec' },
      { label: 'Per Strategy', value: '1,000' },
      { label: 'SC Model', value: 'Bayesian' },
      { label: 'Candidates', value: '~78' },
    ],
    color: '#FF8700',
    icon: '◇',
  },
  {
    id: 5,
    phase: 'Phase 5',
    title: 'Validation & Output',
    description: 'Rolling temporal validation ensures zero data leakage — each fold trains only on past seasons and predicts future ones. Strategy accuracy improves steadily as training data grows.',
    details: [
      'Expanding-window validation: 2022→2023, 2022-23→2024, 2022-24→2025',
      'Exact match: does the model recommend the actual race-winning strategy?',
      'Top-5 match: is the actual strategy in the model\'s top 5 recommendations?',
      'Dynamic Time Warping for mid-race stint similarity analysis (silhouette 0.94)',
    ],
    metrics: [
      { label: 'Exact Match', value: '71%' },
      { label: 'Top-5', value: '86%' },
      { label: 'Leakage', value: 'Zero' },
      { label: 'DTW Score', value: '0.94' },
    ],
    color: '#A855F7',
    icon: '✦',
  },
]

export function Methodology() {
  const [activeStep, setActiveStep] = useState(0)
  const step = steps[activeStep]

  return (
    <section id="methodology" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            System Architecture
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            How It Works
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            A five-phase pipeline transforms raw F1 telemetry into actionable pit stop recommendations,
            using signal processing, gradient-boosted trees, and stochastic simulation.
          </p>
        </div>

        {/* Timeline nav */}
        <div className="relative mb-12">
          {/* Connecting line */}
          <div className="absolute top-6 left-0 right-0 h-px bg-f1-border hidden md:block" />
          <div
            className="absolute top-6 left-0 h-px transition-all duration-500 hidden md:block"
            style={{
              width: `${(activeStep / (steps.length - 1)) * 100}%`,
              background: step.color,
            }}
          />

          <div className="grid grid-cols-5 gap-2 md:gap-4">
            {steps.map((s, i) => (
              <button
                key={s.id}
                onClick={() => setActiveStep(i)}
                className="group relative flex flex-col items-center text-center"
              >
                {/* Node */}
                <div
                  className={`relative z-10 w-12 h-12 rounded-full border-2 flex items-center justify-center text-lg transition-all duration-300 ${
                    i === activeStep
                      ? 'scale-110'
                      : i < activeStep
                      ? 'opacity-80'
                      : 'opacity-40 hover:opacity-70'
                  }`}
                  style={{
                    borderColor: i <= activeStep ? s.color : '#2A2A3A',
                    background: i === activeStep ? `${s.color}20` : '#1A1A26',
                  }}
                >
                  <span>{s.icon}</span>
                </div>

                {/* Label */}
                <div className="mt-3">
                  <div
                    className="font-mono text-[10px] uppercase tracking-wider"
                    style={{ color: i <= activeStep ? s.color : '#8A8A9A' }}
                  >
                    {s.phase}
                  </div>
                  <div className={`font-display font-bold text-xs md:text-sm mt-0.5 transition-colors ${
                    i === activeStep ? 'text-white' : 'text-f1-muted'
                  }`}>
                    {s.title}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Active step detail */}
        <div
          className="bg-f1-card border rounded-lg overflow-hidden transition-all duration-300"
          style={{ borderColor: `${step.color}40` }}
        >
          {/* Top accent bar */}
          <div className="h-1" style={{ background: step.color }} />

          <div className="p-6 md:p-8">
            <div className="flex items-start justify-between flex-wrap gap-4 mb-6">
              <div>
                <div className="font-mono text-xs uppercase tracking-wider mb-2" style={{ color: step.color }}>
                  {step.phase} — {step.title}
                </div>
                <p className="font-body text-f1-muted leading-relaxed max-w-2xl">
                  {step.description}
                </p>
              </div>
            </div>

            {/* Metrics row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
              {step.metrics.map((m, i) => (
                <div
                  key={i}
                  className="bg-f1-darker rounded-lg p-3 border border-f1-border"
                >
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">
                    {m.label}
                  </div>
                  <div className="font-display font-black text-xl text-white mt-1">
                    {m.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Detail bullets */}
            <div className="space-y-3">
              {step.details.map((detail, i) => {
                const [prefix, ...rest] = detail.split(' — ')
                const desc = rest.join(' — ')
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div
                      className="mt-2 w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: step.color }}
                    />
                    <div className="font-body text-sm">
                      {desc ? (
                        <>
                          <span className="text-white font-semibold">{prefix}</span>
                          <span className="text-f1-muted"> — {desc}</span>
                        </>
                      ) : (
                        <span className="text-f1-muted">{prefix}</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Navigation arrows */}
        <div className="flex justify-between mt-6">
          <button
            onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
            disabled={activeStep === 0}
            className="font-mono text-sm text-f1-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Previous Phase
          </button>
          <button
            onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
            disabled={activeStep === steps.length - 1}
            className="font-mono text-sm text-f1-muted hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next Phase →
          </button>
        </div>
      </div>
    </section>
  )
}
