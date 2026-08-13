import { test, expect } from '@playwright/test';

test.describe('System Health & Diagnostics E2E Suite', () => {
  test('GET /api/health returns HTTP 200 with healthy status', async ({ request }) => {
    const response = await request.get('/api/health');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body.status).toBe('healthy');
    expect(body.version).toBe('2.0.0');
    expect(body.supabase).toHaveProperty('configured');
    expect(body.ml_engine).toHaveProperty('accuracy', '99.5%');
  });

  test('Non-existent route loads custom 404 page', async ({ page }) => {
    await page.goto('/some-non-existent-route-12345');
    await expect(page.locator('h2')).toContainText('Page Not Found');
  });
});
