import { type ClaimSubmission } from '../types/claim.types';

export interface DemoCase {
  id: string;
  name: string;
  description: string;
  expected_risk_level: "LOW" | "MEDIUM" | "HIGH";
  data: Partial<ClaimSubmission>;
}

// Generate a valid incident date (3 days ago)
const getRecentDate = () => {
  const date = new Date();
  date.setDate(date.getDate() - 3);
  return date.toISOString().split("T")[0];
};

export const demoCases: DemoCase[] = [
  // ============================================
  // LOW RISK CASES (Should be auto-approved)
  // ============================================
  {
    id: "low-1",
    name: "Typical Medical Expense",
    description: "A completely normal claim with typical amounts, trusted device, and standard form filling behavior.",
    expected_risk_level: "LOW",
    data: {
      claim_type: "medical",
      claim_amount: 450,
      incident_date: getRecentDate(),
      claim_description: "Routine checkup and bloodwork following minor illness. Supporting receipts from the clinic are attached.",
      supporting_documents_count: 2,
      form_fill_time: 75,
      session_duration: 120,
      pages_visited: 3,
      is_trusted_device: true,
      device_trust_score: 0.95,
      is_geolocation_anomaly: false,
      geolocation_distance_km: 5
    }
  },
  {
    id: "low-2",
    name: "Minor Auto Collision",
    description: "Standard auto claim with high device trust. Everything appears consistent with user history.",
    expected_risk_level: "LOW",
    data: {
      claim_type: "accident",
      claim_amount: 1250.50,
      incident_date: getRecentDate(),
      claim_description: "Rear-ended at a red light. Minor bumper damage. Exchanged details and got police report.",
      supporting_documents_count: 4,
      form_fill_time: 110,
      session_duration: 200,
      pages_visited: 2,
      is_trusted_device: true,
      device_trust_score: 0.88,
      is_geolocation_anomaly: false,
      geolocation_distance_km: 12
    }
  },

  // ============================================
  // MEDIUM RISK CASES (Should require OTP)
  // ============================================
  {
    id: "med-1",
    name: "Unusual Claim Amount",
    description: "Trusted device but an unusually large claim amount for property damage.",
    expected_risk_level: "MEDIUM",
    data: {
      claim_type: "property_damage",
      claim_amount: 75000,
      incident_date: getRecentDate(),
      claim_description: "Massive water leak flooded the entire basement, destroying furniture and electronics.",
      supporting_documents_count: 5,
      form_fill_time: 90,
      session_duration: 150,
      pages_visited: 4,
      is_trusted_device: true,
      device_trust_score: 0.85,
      is_geolocation_anomaly: false,
      geolocation_distance_km: 0
    }
  },
  {
    id: "med-2",
    name: "New Untrusted Device",
    description: "A normal claim amount, but submitted from a brand new, untrusted device.",
    expected_risk_level: "MEDIUM",
    data: {
      claim_type: "theft",
      claim_amount: 2100,
      incident_date: getRecentDate(),
      claim_description: "Laptop stolen from coffee shop while I was in the restroom.",
      supporting_documents_count: 1,
      form_fill_time: 65,
      session_duration: 300,
      pages_visited: 6,
      is_trusted_device: false,
      device_trust_score: 0.35,
      is_geolocation_anomaly: false,
      geolocation_distance_km: 25
    }
  },

  // ============================================
  // HIGH RISK / FRAUDULENT (Should require Biometric / Flagged)
  // ============================================
  {
    id: "high-1",
    name: "Identity Theft Simulation",
    description: "Extremely far geolocation, highly untrusted device, and rushed session timing (bot-like).",
    expected_risk_level: "HIGH",
    data: {
      claim_type: "theft",
      claim_amount: 8500,
      incident_date: getRecentDate(),
      claim_description: "Multiple valuable items stolen during vacation.",
      supporting_documents_count: 0,
      form_fill_time: 12, // Bot-like speed
      session_duration: 15,
      pages_visited: 1,
      is_trusted_device: false,
      device_trust_score: 0.05,
      is_geolocation_anomaly: true,
      geolocation_distance_km: 2500 // Different continent
    }
  },
  {
    id: "high-2",
    name: "Staged / Exaggerated Incident",
    description: "Round amount, few documents, extremely suspicious/careful session time, untrusted device.",
    expected_risk_level: "HIGH",
    data: {
      claim_type: "accident",
      claim_amount: 15000, // Round amount spike
      incident_date: getRecentDate(),
      claim_description: "Involved in a multi-car collision. Car is completely totaled.",
      supporting_documents_count: 1,
      form_fill_time: 320, // Overly careful
      session_duration: 400,
      pages_visited: 8,
      is_trusted_device: false,
      device_trust_score: 0.15,
      is_geolocation_anomaly: true,
      geolocation_distance_km: 150
    }
  }
];
