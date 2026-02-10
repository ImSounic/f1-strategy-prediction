'use client'

export function Limitations() {
  const limitations = [
    {
      icon: '🌧️',
      title: 'Dry Conditions Only',
      description: 'The model only handles SOFT, MEDIUM, and HARD compounds. No INTERMEDIATE or WET tyre modeling — rain races are excluded from both training and prediction.',
      severity: 'moderate',
    },
    {
      icon: '🔢',
      title: 'Maximum 2-Stop Strategies',
      description: 'Strategy generation is limited to 1-stop and 2-stop options. Some high-degradation races (e.g. 2022 Spain) require 3+ stops, which the system cannot recommend.',
      severity: 'moderate',
    },
    {
      icon: '📍',
      title: 'No Position Modeling',
      description: 'Optimizes total race time only. Does not account for track position, dirty air, undercut/overcut effects, or the strategic value of staying ahead.',
      severity: 'high',
    },
    {
      icon: '🌡️',
      title: 'Static Weather',
      description: 'Weather conditions are set at race start and held constant. Mid-race temperature changes or sudden rain are not simulated within the Monte Carlo framework.',
      severity: 'low',
    },
    {
      icon: '🏎️',
      title: 'No Car-Specific Modeling',
      description: 'All cars are treated identically. In reality, different teams have different tyre management characteristics — a Red Bull might sustain harder stints than a Williams.',
      severity: 'moderate',
    },
    {
      icon: '⚠️',
      title: 'SC Events Are Stochastic',
      description: 'The honest finding: safety car events are fundamentally unpredictable (RF classifier AUC ≈ 0.5). The Bayesian model captures frequency, not timing.',
      severity: 'low',
    },
  ]

  const futureWork = [
    {
      title: 'Live Telemetry Integration',
      description: 'Real-time lap data feeds to update strategy recommendations mid-race.',
      status: 'planned',
    },
    {
      title: 'Reinforcement Learning',
      description: 'Dynamic re-planning agent that adapts to race events as they unfold.',
      status: 'research',
    },
    {
      title: 'Position-Aware Optimization',
      description: 'Incorporate track position, gap to rivals, and undercut timing into strategy scoring.',
      status: 'planned',
    },
    {
      title: 'Wet Weather Modeling',
      description: 'Extend to INTERMEDIATE/WET compounds with crossover-point prediction.',
      status: 'research',
    },
  ]

  const severityColor: Record<string, string> = {
    low: 'border-green-500/30 bg-green-500/5',
    moderate: 'border-yellow-500/30 bg-yellow-500/5',
    high: 'border-orange-500/30 bg-orange-500/5',
  }

  return (
    <section id="limitations" className="py-24 border-b border-f1-border">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-12">
          <div className="font-mono text-xs text-f1-red uppercase tracking-widest mb-3">
            Honest Assessment
          </div>
          <h2 className="font-display font-black text-4xl md:text-5xl tracking-tight">
            Limitations & Future Work
          </h2>
          <p className="font-body text-f1-muted mt-3 max-w-2xl">
            No model is perfect. Understanding the boundaries of the system is as important
            as understanding its capabilities. These are the known limitations and planned improvements.
          </p>
        </div>

        {/* Limitations grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-16">
          {limitations.map((lim, i) => (
            <div
              key={i}
              className={`rounded-lg border p-5 ${severityColor[lim.severity]}`}
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl">{lim.icon}</span>
                <div>
                  <h4 className="font-display font-bold text-sm text-white mb-1">
                    {lim.title}
                  </h4>
                  <p className="font-body text-xs text-f1-muted leading-relaxed">
                    {lim.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Future work */}
        <div>
          <h3 className="font-display font-bold text-2xl mb-6">
            Future Work
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            {futureWork.map((item, i) => (
              <div
                key={i}
                className="bg-f1-card border border-f1-border rounded-lg p-5 flex items-start gap-4"
              >
                <div className={`mt-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                  item.status === 'planned'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                }`}>
                  {item.status}
                </div>
                <div>
                  <h4 className="font-display font-bold text-sm text-white mb-1">
                    {item.title}
                  </h4>
                  <p className="font-body text-xs text-f1-muted">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
