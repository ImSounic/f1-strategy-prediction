'use client'

export function TechStack() {
  const categories = [
    {
      title: 'Data Mining',
      color: '#E10600',
      items: [
        'XGBoost regression + RandomizedSearchCV (60 iterations)',
        'Random Forest classification (safety car probability)',
        'Hierarchical clustering (circuit similarity)',
        'SHAP interpretability (TreeExplainer)',
        'Model comparison: Ridge vs XGBoost vs MLP',
      ],
    },
    {
      title: 'Time Series',
      color: '#0090D0',
      items: [
        'Savitzky-Golay filtering (degradation curve extraction)',
        'Dynamic Time Warping (stint similarity, silhouette 0.94)',
        'Rolling temporal validation (expanding window)',
        'Derivative analysis (DegSlope as regression target)',
        'Event detection (safety car lap identification)',
      ],
    },
    {
      title: 'Simulation',
      color: '#FF8700',
      items: [
        'Monte Carlo strategy simulation (9,000 sims/sec)',
        'Bayesian Beta-Binomial (safety car priors with shrinkage)',
        'Stochastic SC/VSC injection (per-lap Bernoulli draws)',
        'Fuel-adjusted lap time model',
        'Strategy candidate generation (1-stop, 2-stop)',
      ],
    },
    {
      title: 'Engineering',
      color: '#00D2BE',
      items: [
        'FastF1, OpenF1, Jolpica API integration',
        'Manual Pirelli infographic data collection',
        'Config-driven pipeline (YAML)',
        'FastAPI REST backend',
        'Next.js + Vercel deployment',
      ],
    },
  ]

  return (
    <section id="tech" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Technical Implementation
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Methods & Stack
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {categories.map((cat, i) => (
            <div
              key={i}
              className="bg-f1-card border border-f1-border rounded-lg p-6 relative overflow-hidden"
            >
              {/* Accent stripe */}
              <div
                className="absolute left-0 top-0 bottom-0 w-1"
                style={{ background: cat.color }}
              />

              <h3 className="font-display font-bold text-xl mb-4 pl-4" style={{ color: cat.color }}>
                {cat.title}
              </h3>

              <ul className="space-y-2 pl-4">
                {cat.items.map((item, j) => (
                  <li key={j} className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: cat.color }} />
                    <span className="font-body text-sm text-f1-muted">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Tech badges */}
        <div className="mt-12 flex flex-wrap gap-3 justify-center">
          {[
            'Python 3.10+', 'XGBoost', 'scikit-learn', 'SHAP', 'SciPy',
            'FastF1', 'FastAPI', 'Next.js', 'TypeScript', 'Tailwind CSS',
            'Recharts', 'Vercel', 'Render',
          ].map(tech => (
            <span
              key={tech}
              className="font-mono text-xs px-3 py-1.5 border border-f1-border rounded-full text-f1-muted hover:border-f1-red hover:text-f1-red transition-colors cursor-default"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
