'use client'

export function Hero() {
  return (
    <section className="relative min-h-[90vh] flex items-center overflow-hidden">
      {/* Background grid */}
      <div className="absolute inset-0 grid-bg opacity-50" />

      {/* Red gradient accent */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-f1-red/5 rounded-full blur-[150px]" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-f1-red/3 rounded-full blur-[120px]" />

      <div className="relative max-w-7xl mx-auto px-6 pt-24">
        <div className="max-w-4xl">
          {/* Tag */}
          <div className="animate-fade-in-up stagger-1 inline-flex items-center gap-2 px-3 py-1 rounded-full border border-f1-border bg-f1-card/50 mb-8">
            <span className="w-2 h-2 rounded-full bg-f1-red animate-pulse" />
            <span className="text-xs font-mono text-f1-muted uppercase tracking-wider">
              University of Twente · Data Science
            </span>
          </div>

          {/* Headline */}
          <h1 className="animate-fade-in-up stagger-2 font-display font-black text-5xl md:text-7xl lg:text-8xl leading-[0.9] tracking-tight mb-6">
            F1 Race
            <br />
            Strategy
            <br />
            <span className="text-f1-red">Optimizer</span>
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
              onClick={() => document.getElementById('circuits')?.scrollIntoView({ behavior: 'smooth' })}
              className="group px-8 py-3 bg-f1-red text-white font-display font-bold text-sm uppercase tracking-wider hover:bg-red-700 transition-all glow-red rounded-sm"
            >
              Explore Circuits
              <span className="inline-block ml-2 group-hover:translate-x-1 transition-transform">→</span>
            </button>
            <button
              onClick={() => document.getElementById('validation')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-3 border border-f1-border text-f1-light font-display font-bold text-sm uppercase tracking-wider hover:border-f1-red hover:text-f1-red transition-all rounded-sm"
            >
              View Results
            </button>
          </div>
        </div>

        {/* Pipeline visualization */}
        <div className="animate-fade-in-up stagger-5 mt-20 hidden lg:block">
          <div className="flex items-center gap-1">
            {[
              { label: 'Data Ingestion', detail: '4 APIs · 92 races', color: 'bg-blue-500' },
              { label: 'Feature Engineering', detail: 'Savitzky-Golay', color: 'bg-teal-500' },
              { label: 'XGBoost Model', detail: 'MAE 0.079s', color: 'bg-f1-red' },
              { label: 'Monte Carlo', detail: '9K sims/sec', color: 'bg-orange-500' },
              { label: 'Strategy', detail: '71% accuracy', color: 'bg-purple-500' },
            ].map((step, i) => (
              <div key={i} className="flex items-center">
                <div className={`${step.color}/10 border border-${step.color.replace('bg-', '')}/20 rounded px-4 py-3`}
                     style={{ borderColor: `color-mix(in srgb, currentColor 20%, transparent)` }}>
                  <div className="font-mono text-xs text-f1-muted uppercase tracking-wider">{step.label}</div>
                  <div className="font-display font-bold text-sm text-white mt-1">{step.detail}</div>
                </div>
                {i < 4 && (
                  <div className="text-f1-border mx-2 font-mono">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
