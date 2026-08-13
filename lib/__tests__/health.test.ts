import { describe, it, expect } from 'vitest';
import { GET } from '@/app/api/health/route';

describe('System Health Check API Route Unit Test', () => {
  it('should return valid JSON response with status healthy', async () => {
    const response = await GET();
    expect(response.status).toBe(200);

    const data = await response.json();
    expect(data.status).toBe('healthy');
    expect(data.version).toBe('2.0.0');
    expect(data.ml_engine.accuracy).toBe('99.5%');
  });
});
