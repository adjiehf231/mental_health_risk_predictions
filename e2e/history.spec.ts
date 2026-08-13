import { test, expect } from '@playwright/test';

test.describe('Assessment History E2E Suite', () => {
  test('History page renders table controls and CSV export button', async ({ page }) => {
    await page.goto('/history');

    // Assert main heading
    await expect(page.locator('h1')).toContainText('Assessment History Log');

    // Assert CSV export button
    const exportButton = page.getByRole('button', { name: /Export CSV Report/i });
    await expect(exportButton).toBeVisible();
  });
});
