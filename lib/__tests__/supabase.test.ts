import { describe, it, expect } from 'vitest';
import { isSupabaseConfigured, fetchAssessmentHistory } from '../supabase';

describe('Supabase Client & Fallback Utility', () => {
  it('should accurately detect Supabase configuration state', () => {
    expect(typeof isSupabaseConfigured).toBe('boolean');
  });

  it('should return empty array or fallback array without throwing error when fetching history', async () => {
    const history = await fetchAssessmentHistory();
    expect(Array.isArray(history)).toBe(true);
  });
});
