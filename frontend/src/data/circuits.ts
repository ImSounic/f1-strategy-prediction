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
    key: "albert_park", name: "Albert Park", country: "AU",
    totalLaps: 58, pitLoss: 22.0, scProbability: 0.67, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 1, grip: 3, traction: 3, braking: 3, lateral: 3, stress: 3, downforce: 2, evolution: 4 },
  },
  {
    key: "suzuka", name: "Suzuka International Racing Course", country: "JP",
    totalLaps: 53, pitLoss: 27.0, scProbability: 0.54, compounds: "C1/C2/C3",
    characteristics: { abrasiveness: 3, grip: 3, traction: 3, braking: 3, lateral: 4, stress: 4, downforce: 4, evolution: 2 },
  },
  {
    key: "monaco", name: "Circuit de Monaco", country: "MC",
    totalLaps: 78, pitLoss: 20.0, scProbability: 0.54, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 1, traction: 4, braking: 3, lateral: 1, stress: 2, downforce: 5, evolution: 3 },
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
    key: "monza", name: "Autodromo Nazionale Monza", country: "IT",
    totalLaps: 53, pitLoss: 25.0, scProbability: 0.42, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 2, grip: 2, traction: 4, braking: 4, lateral: 1, stress: 2, downforce: 1, evolution: 2 },
  },
  {
    key: "singapore", name: "Marina Bay Street Circuit", country: "SG",
    totalLaps: 62, pitLoss: 30.0, scProbability: 0.54, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 2, traction: 4, braking: 4, lateral: 2, stress: 3, downforce: 4, evolution: 4 },
  },
  {
    key: "interlagos", name: "Autódromo José Carlos Pace", country: "BR",
    totalLaps: 71, pitLoss: 22.0, scProbability: 0.78, compounds: "C2/C3/C4",
    characteristics: { abrasiveness: 3, grip: 2, traction: 4, braking: 3, lateral: 3, stress: 3, downforce: 3, evolution: 3 },
  },
  {
    key: "yas_marina", name: "Yas Marina Circuit", country: "AE",
    totalLaps: 58, pitLoss: 23.0, scProbability: 0.29, compounds: "C3/C4/C5",
    characteristics: { abrasiveness: 1, grip: 3, traction: 3, braking: 3, lateral: 2, stress: 2, downforce: 3, evolution: 3 },
  },
];
