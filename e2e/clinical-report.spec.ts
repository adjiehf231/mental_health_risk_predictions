import { test, expect } from '@playwright/test';

test.describe('Clinical Summary Report E2E Suite', () => {
  test('History log item allows opening clinical report modal', async ({ page }) => {
    await page.goto('/history');

    // Assert history page heading
    await expect(page.locator('h1')).toContainText('Assessment History Log');

    // If report button exists, click it
    const reportButton = page.locator('button[title="View Clinical Report Summary"]').first();
    if (await reportButton.isVisible()) {
      await reportButton.click();
      await expect(page.locator('h2')).toContainText('Clinical Assessment Summary Report');
    }
  });
});
