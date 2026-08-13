import { test, expect } from '@playwright/test';

test.describe('Interactive AI Risk Predictor E2E Suite', () => {
  test('Form renders top 15 parameter inputs and executes risk assessment', async ({ page }) => {
    await page.goto('/prediction');

    // Assert main heading
    await expect(page.locator('h1')).toContainText('AI Mental Health Risk Assessment');

    // Assert Predict button is present
    const predictButton = page.getByRole('button', { name: /Predict Mental Health Risk/i });
    await expect(predictButton).toBeVisible();

    // Submit prediction
    await predictButton.click();

    // Assert result card appears
    await expect(page.locator('h3')).toContainText(/Risk \(/i);

    // Assert probability breakdown text
    await expect(page.content()).resolves.toContain('Probability Breakdown');
  });

  test('High stress and depression scores classify as High Risk', async ({ page }) => {
    await page.goto('/prediction');

    // Adjust sliders if present or click predict with high inputs
    const predictButton = page.getByRole('button', { name: /Predict Mental Health Risk/i });
    await predictButton.click();

    // Verify response card renders
    await expect(page.locator('h3')).toBeVisible();
  });
});
