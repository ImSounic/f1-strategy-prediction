'use client'

import { useState, useEffect } from 'react'

function AnimatedNumber({ target, suffix = '', duration = 2000 }: { target: number; suffix?: string; duration?: number }) {
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    const steps = 60
    const increment = target / steps
    const stepTime = duration / steps
    let step = 0

    const timer = setInterval(() => {
      step++
      if (step >= steps) {
        setCurrent(target)
        clearInterval(timer)
      } else {
        setCurrent(Math.round(increment * step * 10) / 10)
      }
    }, stepTime)

    return () => clearInterval(timer)
  }, [target, duration])

  return <>{current % 1 === 0 ? current.toFixed(0) : current.toFixed(current < 1 ? 3 : 1)}{suffix}</>
}

export function Hero() {
  const [loaded, setLoaded] = useState(false)
  useEffect(() => setLoaded(true), [])

  return (
    <section className="relative min-h-[90vh] flex items-center overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 grid-bg opacity-50" />

      {/* Animated racing line SVG */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-[0.06]" viewBox="0 0 1440 900" preserveAspectRatio="none">
        <path
          d="M0,450 C200,350 350,550 500,400 S700,200 900,350 S1100,550 1300,300 L1440,400"
          fill="none"
          stroke="#E10600"
          strokeWidth="3"
          className="racing-line-path"
        />
        <path
          d="M0,500 C180,420 320,580 480,430 S680,250 880,380 S1080,570 1280,340 L1440,430"
          fill="none"
          stroke="#0090D0"
          strokeWidth="2"
          className="racing-line-path-2"
          strokeDasharray="8,6"
        />
      </svg>

      {/* Gradient accents */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-f1-red/5 rounded-full blur-[150px]" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-f1-red/3 rounded-full blur-[120px]" />
      <div className="absolute top-1/3 left-1/2 w-[300px] h-[300px] bg-blue-500/3 rounded-full blur-[100px]" />

      <div className="relative max-w-7xl mx-auto px-6 pt-24">
        <div className="max-w-4xl">
          {/* Tag */}
          <div className="animate-fade-in-up stagger-1 inline-flex items-center gap-2 px-3 py-1 rounded-full border border-f1-border bg-f1-card/50 mb-8">
            <span className="w-2 h-2 rounded-full bg-f1-red animate-pulse" />
            <span className="text-xs font-mono text-f1-muted uppercase tracking-wider">
              University of Twente · Data Science Module
            </span>
          </div>

          {/* Headline */}
          <h1 className="animate-fade-in-up stagger-2 font-display font-black text-5xl md:text-7xl lg:text-8xl leading-[0.9] tracking-tight mb-6">
            F1 Race
            <br />
            Strategy
            <br />
            <span className="text-f1-red relative">
              Optimizer
              <span className="absolute -bottom-2 left-0 w-full h-1 bg-gradient-to-r from-f1-red to-transparent" />
            </span>
          </h1>

          {/* Subheading */}
          <p className="animate-fade-in-up stagger-3 font-body text-xl md:text-2xl text-f1-muted max-w-2xl leading-relaxed mb-12">
            Monte Carlo simulation meets machine learning. Predicting optimal
            pit stop strategies with{' '}
            <span className="text-white font-semibold">71% accuracy</span> on
            unseen 2025 races.
          </p>

          {/* CTA */}
          <div className="animate-fade-in-up stagger-4 flex flex-wrap gap-4">
            <button
              onClick={() => document.getElementById('methodology')?.scrollIntoView({ behavior: 'smooth' })}
              className="group px-8 py-3 bg-f1-red text-white font-display font-bold text-sm uppercase tracking-wider hover:bg-red-700 transition-all glow-red rounded-sm"
            >
              How It Works
              <span className="inline-block ml-2 group-hover:translate-x-1 transition-transform">→</span>
            </button>
            <button
              onClick={() => document.getElementById('circuits')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-3 border border-f1-border text-f1-light font-display font-bold text-sm uppercase tracking-wider hover:border-f1-red hover:text-f1-red transition-all rounded-sm"
            >
              Explore Circuits
            </button>
            <button
              onClick={() => document.getElementById('validation')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-3 border border-f1-border text-f1-light font-display font-bold text-sm uppercase tracking-wider hover:border-f1-red hover:text-f1-red transition-all rounded-sm"
            >
              View Results
            </button>
          </div>
        </div>

        {/* Animated Key Metrics */}
        <div className="animate-fade-in-up stagger-5 mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { value: 71, suffix: '%', label: 'Strategy Accuracy', detail: 'unseen 2025 races' },
            { value: 0.079, suffix: 's', label: 'Model MAE', detail: 'per-lap prediction' },
            { value: 92, suffix: '', label: 'Races Analyzed', detail: '2022–2025 seasons' },
            { value: 9000, suffix: '', label: 'Sims / Second', detail: 'Monte Carlo speed' },
          ].map((m, i) => (
            <div key={i} className="bg-f1-card/60 backdrop-blur-sm border border-f1-border rounded-lg p-4 hover:border-f1-red/30 transition-colors">
              <div className="font-display font-black text-3xl md:text-4xl text-white">
                {loaded ? (
                  <AnimatedNumber target={m.value} suffix={m.suffix} duration={1800 + i * 200} />
                ) : (
                  '—'
                )}
              </div>
              <div className="font-display font-bold text-sm text-f1-muted mt-1">{m.label}</div>
              <div className="font-mono text-xs text-f1-border">{m.detail}</div>
            </div>
          ))}
        </div>

        {/* Pipeline visualization */}
        <div className="animate-fade-in-up stagger-5 mt-10 hidden lg:block">
          <div className="flex items-center gap-1">
            {[
              { label: 'Data Ingestion', detail: '4 APIs · 92 races', color: '#0090D0' },
              { label: 'Feature Engineering', detail: 'Savitzky-Golay', color: '#00D2BE' },
              { label: 'XGBoost Model', detail: 'MAE 0.079s', color: '#E10600' },
              { label: 'Monte Carlo', detail: '1K sims/strategy', color: '#FF8700' },
              { label: 'Strategy Output', detail: '71% accuracy', color: '#A855F7' },
            ].map((step, i) => (
              <div key={i} className="flex items-center">
                <div
                  className="rounded px-4 py-3 border transition-all hover:scale-105 cursor-default"
                  style={{
                    borderColor: `${step.color}30`,
                    background: `${step.color}08`,
                  }}
                >
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">{step.label}</div>
                  <div className="font-display font-bold text-sm text-white mt-1">{step.detail}</div>
                </div>
                {i < 4 && (
                  <div className="text-f1-border mx-2 font-mono text-lg">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
