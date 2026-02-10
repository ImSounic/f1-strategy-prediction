const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ApiCircuit {
  circuit_key: string
  circuit_name: string
  season: number
  total_laps: number
  pit_loss_seconds: number
  sc_probability: number
  compounds: string
  characteristics: Record<string, number>
}

export interface ApiStrategyResult {
  strategy_name: string
  compound_sequence: string
  num_stops: number
  median_time: number
  mean_time: number
  std_time: number
  p5_time: number
  p95_time: number
  mean_sc_events: number
}

export interface SimulationResponse {
  circuit_key: string
  circuit_name: string
  season: number
  n_sims: number
  n_strategies: number
  elapsed_seconds: number
  rankings: ApiStrategyResult[]
}

export async function fetchCircuits(season: number = 2025): Promise<ApiCircuit[]> {
  const res = await fetch(`${API_URL}/circuits/${season}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function simulate(
  circuitKey: string,
  season: number = 2025,
  nSims: number = 1000,
  weather?: {
    track_temp?: number
    air_temp?: number
    humidity?: number
    wind_speed?: number
  }
): Promise<SimulationResponse> {
  const res = await fetch(`${API_URL}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      circuit_key: circuitKey,
      season,
      n_sims: nSims,
      ...weather && { weather_override: weather },
    }),
  })
  if (!res.ok) throw new Error(`Simulation failed: ${res.status}`)
  return res.json()
}

export async function fetchValidation() {
  const res = await fetch(`${API_URL}/validation`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/`, { signal: AbortSignal.timeout(3000) })
    return res.ok
  } catch {
    return false
  }
}
