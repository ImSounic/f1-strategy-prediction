export interface StrategyResult {
  rank: number;
  name: string;
  compounds: string;
  stops: number;
  medianTime: number;
  meanTime: number;
  stdTime: number;
  p5: number;
  p95: number;
  delta: number;
  scEvents: number;
}

export interface CircuitStrategy {
  circuit: string;
  circuitName: string;
  season: number;
  nSims: number;
  strategies: StrategyResult[];
}

export const validationData = {
  folds: [
    { label: "2022 → 2023", trainStints: 967, cvMae: 0.1047, exactMatch: 40, top5Match: 50 },
    { label: "2022-23 → 2024", trainStints: 1982, cvMae: 0.0867, exactMatch: 52, top5Match: 71 },
    { label: "2022-24 → 2025", trainStints: 3006, cvMae: 0.0794, exactMatch: 71, top5Match: 86 },
  ],
  models: [
    { name: "Ridge (Baseline)", mae: 0.0777, std: 0.013, time: 1.7 },
    { name: "XGBoost (Primary)", mae: 0.0755, std: 0.012, time: 3.2 },
    { name: "MLP (Neural Net)", mae: 0.0848, std: 0.015, time: 35.9 },
  ],
  shapFeatures: [
    { name: "MeanHumidity", importance: 0.0134 },
    { name: "MeanWindSpeed", importance: 0.0089 },
    { name: "asphalt_grip", importance: 0.0078 },
    { name: "TrackTempRange", importance: 0.0047 },
    { name: "MeanTrackTemp", importance: 0.0046 },
    { name: "traction_demand", importance: 0.0043 },
    { name: "MeanAirTemp", importance: 0.0040 },
    { name: "StintLength", importance: 0.0039 },
    { name: "track_evolution", importance: 0.0016 },
    { name: "StintNumber", importance: 0.0016 },
  ],
};
