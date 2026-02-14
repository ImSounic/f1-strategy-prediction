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
    key: "bahrain", name: "Bahrain International Circuit", country: "BH",
    totalLaps: 57, pitLoss: 23.5, scProbability: 0.54, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 2, traction: 3, braking: 3, lateral: 2, stress: 3, downforce: 2, evolution: 3 },
  },
  {
    key: "jeddah", name: "Jeddah Corniche Circuit", country: "SA",
    totalLaps: 50, pitLoss: 25.0, scProbability: 0.78, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 1, grip: 3, traction: 4, braking: 2, lateral: 3, stress: 4, downforce: 1, evolution: 4 },
  },
  {
    key: "albert_park", name: "Albert Park Circuit", country: "AU",
    totalLaps: 58, pitLoss: 22.0, scProbability: 0.67, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 1, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 2, evolution: 4 },
  },
  {
    key: "shanghai", name: "Shanghai International Circuit", country: "CN",
    totalLaps: 56, pitLoss: 24.0, scProbability: 0.42, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 2, traction: 4, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 3 },
  },
  {
    key: "suzuka", name: "Suzuka International Racing Course", country: "JP",
    totalLaps: 53, pitLoss: 27.0, scProbability: 0.54, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 3, lateral: 4, stress: 4, downforce: 4, evolution: 2 },
  },
  {
    key: "miami", name: "Miami International Autodrome", country: "US",
    totalLaps: 57, pitLoss: 24.0, scProbability: 0.67, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 2, evolution: 4 },
  },
  {
    key: "imola", name: "Autodromo Enzo e Dino Ferrari", country: "IT",
    totalLaps: 63, pitLoss: 25.0, scProbability: 0.54, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "monaco", name: "Circuit de Monaco", country: "MC",
    totalLaps: 78, pitLoss: 20.0, scProbability: 0.54, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 1, traction: 4, braking: 3, lateral: 1, stress: 2, downforce: 5, evolution: 3 },
  },
  {
    key: "montreal", name: "Circuit Gilles Villeneuve", country: "CA",
    totalLaps: 70, pitLoss: 22.0, scProbability: 0.78, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 2, grip: 2, traction: 3, braking: 4, lateral: 2, stress: 3, downforce: 2, evolution: 3 },
  },
  {
    key: "barcelona", name: "Circuit de Barcelona-Catalunya", country: "ES",
    totalLaps: 66, pitLoss: 22.0, scProbability: 0.29, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 2, lateral: 4, stress: 4, downforce: 3, evolution: 2 },
  },
  {
    key: "spielberg", name: "Red Bull Ring", country: "AT",
    totalLaps: 71, pitLoss: 21.0, scProbability: 0.42, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 3, traction: 4, braking: 3, lateral: 2, stress: 2, downforce: 1, evolution: 2 },
  },
  {
    key: "silverstone", name: "Silverstone Circuit", country: "GB",
    totalLaps: 52, pitLoss: 22.0, scProbability: 0.67, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 2, braking: 2, lateral: 4, stress: 4, downforce: 4, evolution: 2 },
  },
  {
    key: "spa", name: "Circuit de Spa-Francorchamps", country: "BE",
    totalLaps: 44, pitLoss: 25.0, scProbability: 0.54, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "hungaroring", name: "Hungaroring", country: "HU",
    totalLaps: 70, pitLoss: 22.0, scProbability: 0.42, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 2, traction: 4, braking: 3, lateral: 3, stress: 3, downforce: 4, evolution: 3 },
  },
  {
    key: "zandvoort", name: "Circuit Zandvoort", country: "NL",
    totalLaps: 72, pitLoss: 20.0, scProbability: 0.54, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 2, grip: 3, traction: 3, braking: 2, lateral: 4, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "monza", name: "Autodromo Nazionale Monza", country: "IT",
    totalLaps: 53, pitLoss: 25.0, scProbability: 0.42, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 2, traction: 4, braking: 4, lateral: 1, stress: 2, downforce: 1, evolution: 2 },
  },
  {
    key: "baku", name: "Baku City Circuit", country: "AZ",
    totalLaps: 51, pitLoss: 27.0, scProbability: 0.78, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 1, grip: 2, traction: 4, braking: 4, lateral: 1, stress: 3, downforce: 1, evolution: 4 },
  },
  {
    key: "singapore", name: "Marina Bay Street Circuit", country: "SG",
    totalLaps: 62, pitLoss: 30.0, scProbability: 0.54, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 2, traction: 4, braking: 4, lateral: 2, stress: 3, downforce: 4, evolution: 4 },
  },
  {
    key: "cota", name: "Circuit of the Americas", country: "US",
    totalLaps: 56, pitLoss: 23.0, scProbability: 0.54, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 2 },
  },
  {
    key: "mexico", name: "Autódromo Hermanos Rodríguez", country: "MX",
    totalLaps: 71, pitLoss: 22.0, scProbability: 0.54, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 2, traction: 4, braking: 3, lateral: 2, stress: 2, downforce: 3, evolution: 3 },
  },
  {
    key: "interlagos", name: "Autódromo José Carlos Pace", country: "BR",
    totalLaps: 71, pitLoss: 22.0, scProbability: 0.78, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 2, traction: 4, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 3 },
  },
  {
    key: "las_vegas", name: "Las Vegas Strip Circuit", country: "US",
    totalLaps: 50, pitLoss: 24.0, scProbability: 0.67, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 1, grip: 2, traction: 3, braking: 3, lateral: 2, stress: 3, downforce: 2, evolution: 4 },
  },
  {
    key: "lusail", name: "Lusail International Circuit", country: "QA",
    totalLaps: 57, pitLoss: 24.0, scProbability: 0.42, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 4, downforce: 3, evolution: 3 },
  },
  {
    key: "yas_marina", name: "Yas Marina Circuit", country: "AE",
    totalLaps: 58, pitLoss: 23.0, scProbability: 0.29, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 3, traction: 3, braking: 3, lateral: 2, stress: 2, downforce: 3, evolution: 3 },
  },
];