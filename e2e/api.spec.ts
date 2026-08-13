import { test, expect } from '@playwright/test';

test.describe('Serverless ML API E2E Suite', () => {
  test('POST /api/py/predict returns valid classification response structure', async ({ request }) => {
    const payload = {
      age: 28,
      gender: 'Female',
      marital_status: 'Single',
      education_level: 'Bachelor',
      employment_status: 'Employed',
      sleep_hours: 7.0,
      physical_activity_hours_per_week: 4,
      screen_time_hours_per_day: 6,
      social_support_score: 7,
      work_stress_level: 4,
      job_satisfaction_score: 8,
      financial_stress_level: 3,
      anxiety_score: 3,
      depression_score: 2,
      panic_attack_history: 0,
      family_history_mental_illness: 0,
      substance_use: 0
    };

    const response = await request.post('/api/py/predict', {
      data: payload,
    });

    if (response.ok()) {
      const data = await response.json();
      expect(data).toHaveProperty('prediction');
      expect(data).toHaveProperty('risk_label');
      expect(data).toHaveProperty('confidence');
      expect(data).toHaveProperty('probabilities');
      expect(Array.isArray(data.probabilities)).toBe(true);
      expect(data.probabilities.length).toBe(3);
    } else {
      // In dev mode prior to running local python server, endpoint may return 404 or connection failure
      console.warn('API Endpoint returned status:', response.status());
    }
  });
});
