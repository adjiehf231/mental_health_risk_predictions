import { describe, it, expect } from 'vitest';
import { PatientAssessmentInput } from '../types';

describe('Data Type Contracts', () => {
  it('should validate patient assessment input object shape', () => {
    const sampleInput: PatientAssessmentInput = {
      age: 25,
      gender: 'Female',
      marital_status: 'Single',
      education_level: 'Bachelor',
      employment_status: 'Employed',
      sleep_hours: 8,
      physical_activity_hours_per_week: 5,
      screen_time_hours_per_day: 4,
      social_support_score: 8,
      work_stress_level: 3,
      job_satisfaction_score: 9,
      financial_stress_level: 2,
      anxiety_score: 2,
      depression_score: 1,
      panic_attack_history: 0,
      family_history_mental_illness: 0,
      substance_use: 0,
    };

    expect(sampleInput.age).toBeGreaterThan(0);
    expect(sampleInput.gender).toBe('Female');
    expect(sampleInput.anxiety_score).toBeLessThanOrEqual(10);
  });
});
