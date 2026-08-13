import { test, expect } from '@playwright/test';

test.describe('EDA Dashboard E2E Suite', () => {
  test('Dashboard loads stats counters and Recharts visual elements', async ({ page }) => {
    await page.goto('/dashboard');

    // Assert main heading
    await expect(page.locator('h1')).toContainText('Dataset EDA & Statistical Insights');

    // Assert stats counters
    await expect(page.content()).resolves.toContain('25,000');
    await expect(page.content()).resolves.toContain('Mental Health Risk Distribution');
  });
});
