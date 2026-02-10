export interface Circuit {
  key: string;
  name: string;
  country: string;
  totalLaps: number;
  pitLoss: number;
  scProbability: number;
  compounds: string;
  characteristics: Record<string, number>;
}

export const circuits: Circuit[] = [
  {
    key: "albert_park", name: "Australian Grand Prix", country: "AU",
    totalLaps: 58, pitLoss: 23.0, scProbability: 0.65, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 3 },
  },
  {
    key: "shanghai", name: "Chinese Grand Prix", country: "CN",
    totalLaps: 56, pitLoss: 24.0, scProbability: 0.54, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "suzuka", name: "Japanese Grand Prix", country: "JP",
    totalLaps: 53, pitLoss: 27.0, scProbability: 0.53, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 4, grip: 4, traction: 4, braking: 4, lateral: 5, stress: 5, downforce: 5, evolution: 3 },
  },
  {
    key: "bahrain", name: "Bahrain Grand Prix", country: "BH",
    totalLaps: 57, pitLoss: 23.5, scProbability: 0.53, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 4, lateral: 3, stress: 4, downforce: 3, evolution: 3 },
  },
  {
    key: "jeddah", name: "Saudi Arabian Grand Prix", country: "SA",
    totalLaps: 50, pitLoss: 25.0, scProbability: 0.78, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 2, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "miami", name: "Miami Grand Prix", country: "US",
    totalLaps: 57, pitLoss: 24.5, scProbability: 0.53, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "imola", name: "Emilia Romagna Grand Prix", country: "IT",
    totalLaps: 63, pitLoss: 25.0, scProbability: 0.6, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 4, evolution: 4 },
  },
  {
    key: "monaco", name: "Monaco Grand Prix", country: "MC",
    totalLaps: 78, pitLoss: 20.0, scProbability: 0.53, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 1, traction: 3, braking: 3, lateral: 3, stress: 2, downforce: 5, evolution: 4 },
  },
  {
    key: "barcelona", name: "Spanish Grand Prix", country: "ES",
    totalLaps: 66, pitLoss: 24.0, scProbability: 0.4, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "montreal", name: "Canadian Grand Prix", country: "CA",
    totalLaps: 70, pitLoss: 22.0, scProbability: 0.78, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 2, grip: 2, traction: 3, braking: 4, lateral: 2, stress: 3, downforce: 2, evolution: 3 },
  },
  {
    key: "spielberg", name: "Austrian Grand Prix", country: "AT",
    totalLaps: 71, pitLoss: 21.0, scProbability: 0.53, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 5, lateral: 3, stress: 4, downforce: 2, evolution: 3 },
  },
  {
    key: "silverstone", name: "British Grand Prix", country: "GB",
    totalLaps: 52, pitLoss: 24.5, scProbability: 0.65, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 3, lateral: 5, stress: 4, downforce: 5, evolution: 3 },
  },
  {
    key: "paul_ricard", name: "French Grand Prix", country: "FR",
    totalLaps: 53, pitLoss: 24.0, scProbability: 0.64, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 4, evolution: 2 },
  },
  {
    key: "spa", name: "Belgian Grand Prix", country: "BE",
    totalLaps: 44, pitLoss: 26.0, scProbability: 0.53, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 3, lateral: 4, stress: 4, downforce: 3, evolution: 3 },
  },
  {
    key: "hungaroring", name: "Hungarian Grand Prix", country: "XX",
    totalLaps: 70, pitLoss: 22.5, scProbability: 0.28, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 3, lateral: 4, stress: 4, downforce: 4, evolution: 4 },
  },
  {
    key: "zandvoort", name: "Dutch Grand Prix", country: "NL",
    totalLaps: 72, pitLoss: 21.5, scProbability: 0.65, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 3, lateral: 5, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "monza", name: "Italian Grand Prix", country: "IT",
    totalLaps: 53, pitLoss: 27.0, scProbability: 0.4, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 5, lateral: 2, stress: 3, downforce: 1, evolution: 4 },
  },
  {
    key: "baku", name: "Azerbaijan Grand Prix", country: "AZ",
    totalLaps: 51, pitLoss: 25.5, scProbability: 0.53, compounds: "C4/C5/C6",
    characteristics: { abrasiveness: 1, grip: 2, traction: 3, braking: 4, lateral: 2, stress: 3, downforce: 3, evolution: 3 },
  },
  {
    key: "singapore", name: "Singapore Grand Prix", country: "SG",
    totalLaps: 62, pitLoss: 30.0, scProbability: 0.53, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 2, traction: 4, braking: 4, lateral: 3, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "cota", name: "United States Grand Prix", country: "US",
    totalLaps: 56, pitLoss: 24.0, scProbability: 0.53, compounds: "C1/C3/C4",
    characteristics: { abrasiveness: 4, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "mexico", name: "Mexico City Grand Prix", country: "MX",
    totalLaps: 71, pitLoss: 22.0, scProbability: 0.53, compounds: "C1/C3/C4",
    characteristics: { abrasiveness: 4, grip: 3, traction: 4, braking: 4, lateral: 3, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "interlagos", name: "São Paulo Grand Prix", country: "BR",
    totalLaps: 71, pitLoss: 22.5, scProbability: 0.78, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 4, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 3, evolution: 3 },
  },
  {
    key: "las_vegas", name: "Las Vegas Grand Prix", country: "US",
    totalLaps: 50, pitLoss: 26.0, scProbability: 0.46, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 1, traction: 3, braking: 4, lateral: 2, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "lusail", name: "Qatar Grand Prix", country: "QA",
    totalLaps: 57, pitLoss: 24.0, scProbability: 0.74, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 4, braking: 4, lateral: 4, stress: 4, downforce: 4, evolution: 3 },
  },
  {
    key: "yas_marina", name: "Abu Dhabi Grand Prix", country: "AE",
    totalLaps: 58, pitLoss: 24.0, scProbability: 0.28, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 4, lateral: 3, stress: 3, downforce: 3, evolution: 3 },
  },
];
