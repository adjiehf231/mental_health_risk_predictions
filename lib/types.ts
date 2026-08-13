export interface PatientAssessmentInput {
  age: number;
  gender: string;
  marital_status: string;
  education_level: string;
  employment_status: string;
  sleep_hours: number;
  physical_activity_hours_per_week: number;
  screen_time_hours_per_day: number;
  social_support_score: number;
  work_stress_level: number;
  job_satisfaction_score: number;
  financial_stress_level: number;
  anxiety_score: number;
  depression_score: number;
  panic_attack_history: number;
  family_history_mental_illness: number;
  substance_use: number;
}

export interface PredictionResult {
  prediction: number;
  risk_label: 'Low Risk (0)' | 'Moderate Risk (1)' | 'High Risk (2)';
  confidence: number;
  probabilities: [number, number, number];
  model_used?: string;
  accuracy?: number;
}

export interface AssessmentRecord extends PatientAssessmentInput {
  id?: string;
  user_id?: string;
  risk_level: string;
  confidence: number;
  probabilities: [number, number, number];
  created_at?: string;
}

export interface ModelMetric {
  name: string;
  accuracy: number;
  f1Score: number;
  precision: number;
  recall: number;
  isBest?: boolean;
}
