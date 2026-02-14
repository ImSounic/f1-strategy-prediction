'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Area, ComposedChart,
} from 'recharts'
import { scenarioData } from '@/data/scenarios'

// ── Driver & Team Data ────────────────────────────────────────

const TEAMS: Record<string, { color: string; name: string }> = {
  red_bull: { color: '#3671C6', name: 'Red Bull' },
  mclaren: { color: '#FF8000', name: 'McLaren' },
  ferrari: { color: '#E8002D', name: 'Ferrari' },
  mercedes: { color: '#27F4D2', name: 'Mercedes' },
  aston_martin: { color: '#229971', name: 'Aston Martin' },
  alpine: { color: '#FF87BC', name: 'Alpine' },
  williams: { color: '#64C4FF', name: 'Williams' },
  rb: { color: '#6692FF', name: 'RB' },
  kick_sauber: { color: '#52E252', name: 'Kick Sauber' },
  haas: { color: '#B6BABD', name: 'Haas' },
}

const DRIVERS = [
  { code: 'VER', name: 'Max Verstappen', team: 'red_bull', number: 1 },
  { code: 'PER', name: 'Sergio Perez', team: 'red_bull', number: 11 },
  { code: 'NOR', name: 'Lando Norris', team: 'mclaren', number: 4 },
  { code: 'PIA', name: 'Oscar Piastri', team: 'mclaren', number: 81 },
  { code: 'LEC', name: 'Charles Leclerc', team: 'ferrari', number: 16 },
  { code: 'SAI', name: 'Carlos Sainz', team: 'ferrari', number: 55 },
  { code: 'HAM', name: 'Lewis Hamilton', team: 'mercedes', number: 44 },
  { code: 'RUS', name: 'George Russell', team: 'mercedes', number: 63 },
  { code: 'ALO', name: 'Fernando Alonso', team: 'aston_martin', number: 14 },
  { code: 'STR', name: 'Lance Stroll', team: 'aston_martin', number: 18 },
  { code: 'GAS', name: 'Pierre Gasly', team: 'alpine', number: 10 },
  { code: 'OCO', name: 'Esteban Ocon', team: 'alpine', number: 31 },
  { code: 'ALB', name: 'Alexander Albon', team: 'williams', number: 23 },
  { code: 'SAR', name: 'Logan Sargeant', team: 'williams', number: 2 },
  { code: 'TSU', name: 'Yuki Tsunoda', team: 'rb', number: 22 },
  { code: 'RIC', name: 'Daniel Ricciardo', team: 'rb', number: 3 },
  { code: 'BOT', name: 'Valtteri Bottas', team: 'kick_sauber', number: 77 },
  { code: 'ZHO', name: 'Zhou Guanyu', team: 'kick_sauber', number: 24 },
  { code: 'MAG', name: 'Kevin Magnussen', team: 'haas', number: 20 },
  { code: 'HUL', name: 'Nico Hulkenberg', team: 'haas', number: 27 },
]

const CIRCUITS = [
  { key: 'bahrain', name: 'Bahrain', flag: '🇧🇭' },
  { key: 'jeddah', name: 'Jeddah', flag: '🇸🇦' },
  { key: 'albert_park', name: 'Albert Park', flag: '🇦🇺' },
  { key: 'suzuka', name: 'Suzuka', flag: '🇯🇵' },
  { key: 'shanghai', name: 'Shanghai', flag: '🇨🇳' },
  { key: 'miami', name: 'Miami', flag: '🇺🇸' },
  { key: 'imola', name: 'Imola', flag: '🇮🇹' },
  { key: 'monaco', name: 'Monaco', flag: '🇲🇨' },
  { key: 'montreal', name: 'Montréal', flag: '🇨🇦' },
  { key: 'barcelona', name: 'Barcelona', flag: '🇪🇸' },
  { key: 'spielberg', name: 'Spielberg', flag: '🇦🇹' },
  { key: 'silverstone', name: 'Silverstone', flag: '🇬🇧' },
  { key: 'hungaroring', name: 'Hungaroring', flag: '🇭🇺' },
  { key: 'spa', name: 'Spa', flag: '🇧🇪' },
  { key: 'zandvoort', name: 'Zandvoort', flag: '🇳🇱' },
  { key: 'monza', name: 'Monza', flag: '🇮🇹' },
  { key: 'baku', name: 'Baku', flag: '🇦🇿' },
  { key: 'singapore', name: 'Singapore', flag: '🇸🇬' },
  { key: 'cota', name: 'COTA', flag: '🇺🇸' },
  { key: 'mexico', name: 'Mexico City', flag: '🇲🇽' },
  { key: 'interlagos', name: 'Interlagos', flag: '🇧🇷' },
  { key: 'las_vegas', name: 'Las Vegas', flag: '🇺🇸' },
  { key: 'yas_marina', name: 'Yas Marina', flag: '🇦🇪' },
]

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#ff3333',
  MEDIUM: '#ffd700',
  HARD: '#e0e0e0',
}

// Scenario tab config
const SCENARIO_TABS = [
  { key: 'best',     label: 'Best Case',  icon: '🏆', color: '#10b981' },
  { key: 'median',   label: 'Median',     icon: '📊', color: '#3b82f6' },
  { key: 'worst',    label: 'Worst Case', icon: '⚠️',  color: '#ef4444' },
  { key: 'early_sc', label: 'Early SC',   icon: '🚨', color: '#f59e0b' },
  { key: 'late_sc',  label: 'Late SC',    icon: '🏁', color: '#8b5cf6' },
] as const

type ScenarioKey = typeof SCENARIO_TABS[number]['key']

// ── Sub-Components ────────────────────────────────────────────

function CompoundPill({ compound }: { compound: string }) {
  const bg = COMPOUND_COLORS[compound] || '#888'
  const text = compound === 'HARD' ? '#111' : compound === 'MEDIUM' ? '#111' : '#fff'
  return (
    <span
      className="inline-block px-2.5 py-1 rounded text-xs font-mono font-bold leading-none"
      style={{ backgroundColor: bg, color: text }}
    >
      {compound}
    </span>
  )
}

function DriverCard({
  driver,
  selected,
  onClick,
}: {
  driver: typeof DRIVERS[0]
  selected: boolean
  onClick: () => void
}) {
  const team = TEAMS[driver.team]
  return (
    <button
      onClick={onClick}
      className={`relative flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all text-left w-full ${
        selected
          ? 'border-white/40 bg-white/10 ring-1 ring-white/20'
          : 'border-f1-border bg-f1-darker hover:border-white/20 hover:bg-white/5'
      }`}
    >
      <div
        className="w-1 h-8 rounded-full shrink-0"
        style={{ backgroundColor: team.color }}
      />
      <div className="min-w-0">
        <div className="font-mono text-xs text-f1-muted">{team.name}</div>
        <div className="font-display font-bold text-sm truncate">{driver.name}</div>
      </div>
      <div
        className="ml-auto font-display font-black text-2xl opacity-20 shrink-0"
        style={{ color: team.color }}
      >
        {driver.number}
      </div>
    </button>
  )
}

function GridPositionSlider({
  value,
  onChange,
}: {
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs text-f1-muted">Starting Grid Position</span>
        <span className="font-display font-black text-3xl">
          P{value}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={1}
          max={20}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-2 appearance-none bg-f1-darker rounded-full cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-6 
            [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:rounded-full 
            [&::-webkit-slider-thumb]:bg-f1-red [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white/30
            [&::-webkit-slider-thumb]:shadow-lg"
        />
        <div className="flex justify-between mt-1 font-mono text-[10px] text-f1-muted/50">
          <span>P1</span>
          <span>P5</span>
          <span>P10</span>
          <span>P15</span>
          <span>P20</span>
        </div>
      </div>
    </div>
  )
}

function PositionChart({ 
  positions,
  compounds,
  tyreAges,
  pitLaps,
  scLaps,
  driverColor,
}: { 
  positions: number[]
  compounds: string[]
  tyreAges: number[]
  pitLaps: number[]
  scLaps: number[]
  driverColor: string
}) {
  const data = positions.map((pos, i) => ({
    lap: i + 1,
    position: pos,
    compound: compounds[i] || 'MEDIUM',
    tyreAge: tyreAges[i] || 0,
  }))

  // Stint background bands
  const stintBands: { x1: number; x2: number; compound: string }[] = []
  if (compounds.length > 0) {
    let stintStart = 1
    let currentCompound = compounds[0]
    for (let i = 1; i < compounds.length; i++) {
      if (compounds[i] !== currentCompound) {
        stintBands.push({ x1: stintStart, x2: i + 1, compound: currentCompound })
        stintStart = i + 1
        currentCompound = compounds[i]
      }
    }
    stintBands.push({ x1: stintStart, x2: compounds.length, compound: currentCompound })
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.[0]) return null
    const d = payload[0].payload
    const isPit = pitLaps.includes(d.lap)
    const isSC = scLaps.includes(d.lap)
    const compColor = COMPOUND_COLORS[d.compound] || '#888'
    return (
      <div className="bg-[#1a1a2e] border border-[#333] rounded-lg px-3 py-2 shadow-xl">
        <div className="font-mono text-xs text-f1-muted mb-1">Lap {d.lap}</div>
        <div className="font-display font-bold text-lg" style={{ color: driverColor }}>P{d.position}</div>
        <div className="flex items-center gap-2 mt-1">
          <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: compColor }} />
          <span className="font-mono text-xs">{d.compound} — {d.tyreAge} laps old</span>
        </div>
        {isPit && <div className="font-mono text-xs text-yellow-400 mt-1">⬇ PIT STOP</div>}
        {isSC && <div className="font-mono text-xs text-red-400 mt-1">🚨 Safety Car</div>}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
        
        {stintBands.map((band, i) => {
          const color = COMPOUND_COLORS[band.compound] || '#888'
          return (
            <ReferenceArea
              key={`stint-${i}`}
              x1={band.x1} x2={band.x2}
              fill={color} fillOpacity={0.06}
              ifOverflow="extendDomain"
            />
          )
        })}
        
        <XAxis
          dataKey="lap"
          tick={{ fontSize: 10, fill: '#666' }}
          label={{ value: 'Lap', position: 'insideBottom', offset: -2, fontSize: 11, fill: '#888' }}
        />
        <YAxis
          reversed
          domain={[1, 20]}
          tick={{ fontSize: 10, fill: '#666' }}
          label={{ value: 'Position', angle: -90, position: 'insideLeft', offset: 15, fontSize: 11, fill: '#888' }}
          ticks={[1, 5, 10, 15, 20]}
        />
        <Tooltip content={<CustomTooltip />} />
        
        {scLaps.map(lap => (
          <ReferenceLine
            key={`sc-${lap}`} x={lap}
            stroke="#ff3333" strokeDasharray="3 3" strokeOpacity={0.5}
          />
        ))}
        
        {pitLaps.map(lap => (
          <ReferenceLine
            key={`pit-${lap}`} x={lap}
            stroke="#ffd700" strokeWidth={2} strokeOpacity={0.9}
            label={{ value: 'PIT', position: 'top', fontSize: 9, fill: '#ffd700', fontFamily: 'monospace' }}
          />
        ))}
        
        <Line
          type="stepAfter"
          dataKey="position"
          stroke={driverColor}
          strokeWidth={2.5}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

function TyreChart({ 
  tyreAges, 
  compounds, 
  pitLaps,
  scLaps,
}: { 
  tyreAges: number[]
  compounds: string[]
  pitLaps: number[]
  scLaps: number[]
}) {
  const data = tyreAges.map((age, i) => ({
    lap: i + 1,
    tyreAge: age,
  }))

  return (
    <ResponsiveContainer width="100%" height={100}>
      <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <defs>
          <linearGradient id="simTyreGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis dataKey="lap" tick={{ fontSize: 9, fill: '#666' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#666' }} tickLine={false} axisLine={false} />
        {scLaps.map(lap => (
          <ReferenceLine key={`sc-${lap}`} x={lap} stroke="#ff3333" strokeDasharray="2 2" strokeOpacity={0.5} />
        ))}
        {pitLaps.map(lap => (
          <ReferenceLine key={`pit-${lap}`} x={lap} stroke="#ffd700" strokeWidth={2} strokeOpacity={0.8} />
        ))}
        <Area
          type="monotone" dataKey="tyreAge" stroke="#10b981"
          fill="url(#simTyreGrad)" strokeWidth={1.5}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// ── Race Scenario Panel ──────────────────────────────────────

interface SampleRaceData {
  positions: number[]
  tyre_ages: number[]
  compounds: string[]
  pit_laps: number[]
  sc_laps: number[]
  target_position: number
  narrative: string
}

function ScenarioPanel({
  race,
  driverColor,
  gridPosition,
  tabColor,
}: {
  race: SampleRaceData
  driverColor: string
  gridPosition: number
  tabColor: string
}) {
  const gain = gridPosition - race.target_position
  
  // Build compound sequence display
  const compoundSequence: string[] = []
  if (race.compounds.length > 0) {
    compoundSequence.push(race.compounds[0])
    for (let i = 1; i < race.compounds.length; i++) {
      if (race.compounds[i] !== race.compounds[i - 1]) {
        compoundSequence.push(race.compounds[i])
      }
    }
  }

  return (
    <div className="space-y-4">
      {/* Narrative + result badge */}
      <div className="flex flex-col md:flex-row md:items-start gap-4">
        <div className="flex-1">
          <p className="font-body text-sm text-f1-muted leading-relaxed">
            {race.narrative}
          </p>
        </div>
        <div className="shrink-0 flex items-center gap-3">
          <div 
            className="px-4 py-2 rounded-lg border text-center"
            style={{ borderColor: tabColor + '40', background: tabColor + '15' }}
          >
            <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest">Finish</div>
            <div className="font-display font-black text-2xl" style={{ color: tabColor }}>
              P{race.target_position}
            </div>
            <div className={`font-mono text-xs mt-0.5 ${
              gain > 0 ? 'text-emerald-400' : gain < 0 ? 'text-red-400' : 'text-f1-muted'
            }`}>
              {gain > 0 ? '+' : ''}{gain} pos
            </div>
          </div>
          <div className="flex flex-col items-center gap-1">
            {compoundSequence.map((c, i) => (
              <div key={i} className="flex items-center gap-1">
                {i > 0 && <span className="font-mono text-[9px] text-f1-muted">↓</span>}
                <CompoundPill compound={c} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Position chart */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[10px] text-f1-muted uppercase tracking-widest">
            Position Through Race
          </span>
          <div className="flex items-center gap-3 font-mono text-[10px]">
            {race.pit_laps.length > 0 && (
              <span className="flex items-center gap-1.5">
                <span className="w-4 h-0.5 bg-yellow-400 inline-block" /> PIT
              </span>
            )}
            {race.sc_laps.length > 0 && (
              <span className="flex items-center gap-1.5">
                <span className="w-4 h-0.5 bg-red-500 inline-block opacity-60" /> SC
              </span>
            )}
          </div>
        </div>
        <PositionChart
          positions={race.positions}
          compounds={race.compounds}
          tyreAges={race.tyre_ages}
          pitLaps={race.pit_laps}
          scLaps={race.sc_laps}
          driverColor={driverColor}
        />
      </div>

      {/* Tyre age mini chart */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="font-mono text-[10px] text-f1-muted uppercase tracking-widest">
            Tyre Age
          </span>
          <div className="flex items-center gap-1">
            {compoundSequence.map((c, i) => (
              <span key={i} className="flex items-center gap-1">
                {i > 0 && <span className="font-mono text-[9px] text-f1-muted">→</span>}
                <CompoundPill compound={c} />
              </span>
            ))}
            {race.pit_laps.length > 0 && (
              <span className="font-mono text-[10px] text-f1-muted ml-2">
                Pit L{race.pit_laps.join(', L')}
              </span>
            )}
          </div>
        </div>
        <TyreChart
          tyreAges={race.tyre_ages}
          compounds={race.compounds}
          pitLaps={race.pit_laps}
          scLaps={race.sc_laps}
        />
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────

export function SimulatorView() {
  const [mounted, setMounted] = useState(false)
  const [selectedDriver, setSelectedDriver] = useState('VER')
  const [selectedCircuit, setSelectedCircuit] = useState('bahrain')
  const [gridPosition, setGridPosition] = useState(1)
  const [activeTab, setActiveTab] = useState<ScenarioKey>('median')
  const [detailData, setDetailData] = useState<Record<string, any> | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  // Only render after client-side hydration
  useEffect(() => {
    setMounted(true)
  }, [])

  const driver = DRIVERS.find(d => d.code === selectedDriver)!
  const circuit = CIRCUITS.find(c => c.key === selectedCircuit)!
  const teamColor = TEAMS[driver.team].color

  // Look up pre-computed stats from TS
  const result = useMemo(() => {
    if (!mounted) return null
    if (!scenarioData || typeof scenarioData !== 'object') return null
    const circuitData = scenarioData[selectedCircuit]
    if (!circuitData?.drivers?.[selectedDriver]?.[String(gridPosition)]) {
      return null
    }
    return circuitData.drivers[selectedDriver][String(gridPosition)]
  }, [selectedCircuit, selectedDriver, gridPosition, mounted])

  // Fetch detailed sample_races JSON when circuit changes
  useEffect(() => {
    setDetailData(null)
    setLoadingDetail(true)
    fetch(`/scenarios/${selectedCircuit}.json`)
      .then(res => {
        if (!res.ok) throw new Error('Not found')
        return res.json()
      })
      .then(data => {
        setDetailData(data)
        setLoadingDetail(false)
      })
      .catch(() => {
        setDetailData(null)
        setLoadingDetail(false)
      })
  }, [selectedCircuit])

  // Get sample_races for current combo
  const sampleRaces = useMemo(() => {
    if (!detailData) return null
    return detailData?.drivers?.[selectedDriver]?.[String(gridPosition)] || null
  }, [detailData, selectedDriver, gridPosition])

  // Available tabs (only show SC tabs if data exists)
  const availableTabs = useMemo(() => {
    if (!mounted || !sampleRaces) return SCENARIO_TABS.filter(t => t.key === 'median')
    return SCENARIO_TABS.filter(t => sampleRaces[t.key])
  }, [sampleRaces, mounted])

  // Reset to median tab when combo changes and current tab isn't available
  useEffect(() => {
    if (!availableTabs.find(t => t.key === activeTab)) {
      setActiveTab('median')
    }
  }, [availableTabs, activeTab])

  const activeRace = sampleRaces?.[activeTab] as SampleRaceData | undefined
  const activeTabConfig = SCENARIO_TABS.find(t => t.key === activeTab)!

  // Prevent SSR/static generation - only render on client
  if (!mounted) {
    return (
      <div className="pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-center min-h-[60vh]">
          <div className="text-center">
            <div className="inline-block w-8 h-8 border-3 border-f1-muted border-t-emerald-400 rounded-full animate-spin mb-4" />
            <div className="font-mono text-sm text-f1-muted">Loading simulator...</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        {/* Hero */}
        <div className="mb-12">
          <div className="font-mono text-xs text-emerald-400 uppercase tracking-widest mb-3">
            Multi-Car Race Simulation
          </div>
          <h1 className="font-display font-black text-4xl md:text-5xl tracking-tight mb-3">
            Race Strategy Simulator
          </h1>
          <p className="font-body text-f1-muted max-w-2xl">
            Pick a driver, circuit, and starting position. The simulator runs a full 20-car field
            with overtaking, DRS, safety cars, team orders, and tyre-dependent strategy
            optimization to find the best race strategy.
          </p>
        </div>

        {/* Controls grid */}
        <div className="grid lg:grid-cols-[1fr_320px] gap-8 mb-12">
          {/* Left: Driver selection */}
          <div className="bg-f1-card border border-f1-border rounded-lg p-6">
            <h2 className="font-display font-bold text-lg mb-4">Select Driver</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {DRIVERS.map(d => (
                <DriverCard
                  key={d.code}
                  driver={d}
                  selected={d.code === selectedDriver}
                  onClick={() => setSelectedDriver(d.code)}
                />
              ))}
            </div>
          </div>

          {/* Right: Circuit + Grid Position */}
          <div className="flex flex-col gap-6">
            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <h2 className="font-display font-bold text-lg mb-4">Circuit</h2>
              <select
                value={selectedCircuit}
                onChange={(e) => setSelectedCircuit(e.target.value)}
                className="w-full bg-f1-darker border border-f1-border rounded-lg px-4 py-3 
                  font-mono text-sm text-white appearance-none cursor-pointer
                  focus:outline-none focus:border-emerald-500/50"
              >
                {CIRCUITS.map(c => (
                  <option key={c.key} value={c.key}>
                    {c.flag} {c.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <GridPositionSlider value={gridPosition} onChange={setGridPosition} />
            </div>

            {/* Quick summary */}
            <div
              className="rounded-lg p-4 border"
              style={{ borderColor: teamColor + '40', background: teamColor + '10' }}
            >
              <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
                Simulating
              </div>
              <div className="font-display font-bold text-lg">
                {driver.name}
              </div>
              <div className="font-mono text-xs text-f1-muted">
                P{gridPosition} at {circuit.flag} {circuit.name}
              </div>
            </div>
          </div>
        </div>

        {/* Results */}
        {result ? (
          <div className="space-y-6">
            {/* Strategy recommendation */}
            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                <div>
                  <div className="font-mono text-xs text-emerald-400 uppercase tracking-widest mb-1">
                    Optimal Strategy
                  </div>
                  <h3 className="font-display font-black text-2xl">
                    {result.strategy}
                  </h3>
                  <div className="font-mono text-xs text-f1-muted mt-1">
                    {result.stops}-stop strategy — Pit on{' '}
                    {result.pitLaps.map((l: number) => `L${l}`).join(', ')}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {result.compounds.map((c: string, i: number) => (
                    <span key={i} className="flex items-center gap-1">
                      <CompoundPill compound={c} />
                      {i < result.compounds.length - 1 && (
                        <span className="text-f1-muted font-mono text-xs">→</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>

              {/* Position stats */}
              <div className="grid grid-cols-4 gap-4">
                {[
                  { label: 'Best Case', value: `P${result.bestPos}`, color: '#10b981' },
                  { label: 'Median Finish', value: `P${result.medianPos}`, color: teamColor },
                  { label: 'Mean Finish', value: `P${result.meanPos.toFixed(1)}`, color: '#888' },
                  { label: 'Worst Case', value: `P${result.worstPos}`, color: '#ef4444' },
                ].map(stat => (
                  <div key={stat.label} className="bg-f1-darker border border-f1-border rounded-lg p-4">
                    <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
                      {stat.label}
                    </div>
                    <div className="font-display font-black text-2xl" style={{ color: stat.color }}>
                      {stat.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Scenario tabs + charts */}
            <div className="bg-f1-card border border-f1-border rounded-lg overflow-hidden">
              {/* Tab bar */}
              <div className="flex border-b border-f1-border overflow-x-auto">
                {availableTabs.map(tab => {
                  const isActive = activeTab === tab.key
                  return (
                    <button
                      key={tab.key}
                      onClick={() => setActiveTab(tab.key)}
                      className={`flex items-center gap-2 px-5 py-3.5 font-mono text-sm whitespace-nowrap transition-all border-b-2 ${
                        isActive
                          ? 'text-white border-current'
                          : 'text-f1-muted border-transparent hover:text-white/70 hover:bg-white/5'
                      }`}
                      style={isActive ? { color: tab.color, borderColor: tab.color } : undefined}
                    >
                      <span>{tab.icon}</span>
                      <span>{tab.label}</span>
                      {sampleRaces?.[tab.key] && (
                        <span 
                          className="text-[10px] px-1.5 py-0.5 rounded font-bold"
                          style={{ 
                            backgroundColor: tab.color + '20', 
                            color: tab.color 
                          }}
                        >
                          P{sampleRaces[tab.key].target_position}
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>

              {/* Tab content */}
              <div className="p-6">
                {loadingDetail ? (
                  <div className="text-center py-12">
                    <div className="inline-block w-6 h-6 border-2 border-f1-muted border-t-emerald-400 rounded-full animate-spin mb-3" />
                    <div className="font-mono text-sm text-f1-muted">Loading race scenarios...</div>
                  </div>
                ) : activeRace ? (
                  <ScenarioPanel
                    race={activeRace}
                    driverColor={teamColor}
                    gridPosition={gridPosition}
                    tabColor={activeTabConfig.color}
                  />
                ) : (
                  result.positionTrace?.length > 0 ? (
                    <div>
                      <p className="font-body text-sm text-f1-muted mb-4">
                        Detailed race scenarios loading. Showing summary position trace.
                      </p>
                      <PositionChart
                        positions={result.positionTrace}
                        compounds={[]}
                        tyreAges={[]}
                        pitLaps={result.pitLaps || []}
                        scLaps={[]}
                        driverColor={teamColor}
                      />
                    </div>
                  ) : (
                    <div className="text-center py-8 font-mono text-sm text-f1-muted">
                      No detailed race data available for this scenario.
                    </div>
                  )
                )}
              </div>
            </div>

            {/* Position gain/loss analysis */}
            <div className="bg-f1-card border border-f1-border rounded-lg p-6">
              <h3 className="font-display font-bold text-lg mb-4">Position Analysis</h3>
              <div className="grid md:grid-cols-3 gap-4">
                {(() => {
                  const gain = gridPosition - result.medianPos
                  return (
                    <>
                      <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
                        <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
                          Expected Gain/Loss
                        </div>
                        <div className={`font-display font-black text-2xl ${
                          gain > 0 ? 'text-emerald-400' : gain < 0 ? 'text-red-400' : 'text-f1-muted'
                        }`}>
                          {gain > 0 ? '+' : ''}{gain.toFixed(1)} positions
                        </div>
                        <div className="font-mono text-[10px] text-f1-muted mt-1">
                          P{gridPosition} → P{result.medianPos} (median)
                        </div>
                      </div>
                      <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
                        <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
                          Position Range
                        </div>
                        <div className="font-display font-black text-2xl text-white">
                          P{result.bestPos} – P{result.worstPos}
                        </div>
                        <div className="font-mono text-[10px] text-f1-muted mt-1">
                          5th to 95th percentile
                        </div>
                      </div>
                      <div className="bg-f1-darker border border-f1-border rounded-lg p-4">
                        <div className="font-mono text-[10px] text-f1-muted uppercase tracking-widest mb-1">
                          Strategy Type
                        </div>
                        <div className="font-display font-black text-2xl text-white">
                          {result.stops}-Stop
                        </div>
                        <div className="font-mono text-[10px] text-f1-muted mt-1">
                          {result.compounds.join(' → ')}
                        </div>
                      </div>
                    </>
                  )
                })()}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-f1-card border border-f1-border rounded-lg p-12 text-center">
            <div className="font-display font-bold text-xl mb-2 text-f1-muted">
              Scenario Data Not Yet Available
            </div>
            <p className="font-mono text-sm text-f1-muted max-w-md mx-auto">
              Pre-computed data for this combination hasn&apos;t been generated yet.
            </p>
            <div className="mt-4 font-mono text-xs text-f1-border">
              python -m src.simulation.precompute_scenarios --circuits {selectedCircuit}
            </div>
          </div>
        )}

        {/* Method note */}
        <div className="mt-8 font-mono text-xs text-f1-border leading-relaxed">
          Simulation uses a 20-car field with driver pace offsets, tyre-dependent overtaking (DRS, dirty air),
          blue flags, team orders, and stochastic safety cars. Strategy optimization combines Monte Carlo
          search over 50 simulations with greedy SC reaction. Best/worst cases are actual outcomes from the
          simulations; SC scenarios show representative races with early or late safety car deployments.
        </div>
      </div>
    </div>
  )
}