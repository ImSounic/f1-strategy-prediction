'use client'

export function Footer() {
  return (
    <footer className="py-16 bg-f1-darker">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-3 gap-12">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 bg-f1-red rounded-sm flex items-center justify-center">
                <span className="font-display font-black text-white text-sm">F1</span>
              </div>
              <span className="font-display font-bold text-lg">
                Strategy<span className="text-f1-red">Optimizer</span>
              </span>
            </div>
            <p className="font-body text-sm text-f1-muted leading-relaxed">
              A data-driven system for optimizing Formula 1 pit stop strategies
              using machine learning and Monte Carlo simulation.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-display font-bold text-sm uppercase tracking-wider mb-4">
              Resources
            </h4>
            <div className="space-y-2">
              {[
                ['FastF1 API', 'https://docs.fastf1.dev/'],
                ['OpenF1 API', 'https://openf1.org/'],
                ['Jolpica API', 'https://api.jolpi.ca/ergast/'],
                ['SHAP Documentation', 'https://shap.readthedocs.io/'],
              ].map(([label, url]) => (
                <a
                  key={label}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block font-body text-sm text-f1-muted hover:text-f1-red transition-colors"
                >
                  {label} ↗
                </a>
              ))}
            </div>
          </div>

          {/* Academic */}
          <div>
            <h4 className="font-display font-bold text-sm uppercase tracking-wider mb-4">
              Academic
            </h4>
            <p className="font-body text-sm text-f1-muted leading-relaxed">
              University of Twente<br />
              Data Science Module<br />
              Topics: Data Mining · Feature Extraction from Time Series
            </p>
            <div className="mt-4 font-mono text-xs text-f1-border">
              2022–2025 F1 data · 92 races · 4,250 stints
            </div>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-f1-border flex flex-wrap justify-between items-center gap-4">
          <div className="font-mono text-xs text-f1-border">
            © 2025 F1 Strategy Optimizer
          </div>
          <div className="font-mono text-xs text-f1-border">
            Built with Next.js · FastAPI · XGBoost · Monte Carlo Simulation
          </div>
        </div>
      </div>
    </footer>
  )
}
