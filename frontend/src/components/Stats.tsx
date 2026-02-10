'use client'

export function Stats() {
  const stats = [
    { value: '71%', label: 'Exact Match', sub: '2025 dry races' },
    { value: '86%', label: 'Top-5 Match', sub: 'unseen season' },
    { value: '0.079s', label: 'Model MAE', sub: 'per lap' },
    { value: '9,000', label: 'Simulations/sec', sub: 'Monte Carlo' },
    { value: '92', label: 'Races Analyzed', sub: '2022–2025' },
    { value: '4,250', label: 'Tyre Stints', sub: 'engineered features' },
  ]

  return (
    <section className="border-y border-f1-border bg-f1-dark/50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
          {stats.map((stat, i) => (
            <div key={i} className="text-center">
              <div className="font-display font-black text-2xl md:text-3xl text-white">
                {stat.value}
              </div>
              <div className="font-body text-sm text-f1-muted mt-1">{stat.label}</div>
              <div className="font-mono text-xs text-f1-border mt-0.5">{stat.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
