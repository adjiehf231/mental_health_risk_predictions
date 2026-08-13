import { test, expect } from '@playwright/test';

test.describe('Phase 6 Production Release Readiness E2E Suite', () => {
  test('Complete End-to-End User Journey Walkthrough', async ({ page }) => {
    // 1. Load Homepage
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Mental Health Risk');

    // 2. Navigate to AI Risk Predictor
    await page.getByRole('link', { name: 'AI Predictor' }).first().click();
    await expect(page).toHaveURL(/\/prediction/);

    // Adjust sliders & submit assessment
    const predictButton = page.getByRole('button', { name: /Predict Mental Health Risk/i });
    await expect(predictButton).toBeVisible();
    await predictButton.click();

    // Verify Prediction Card & Probability Breakdown Chart
    await expect(page.locator('h3')).toContainText(/Risk \(/i);
    await expect(page.content()).resolves.toContain('Probability Breakdown');

    // 3. Navigate to EDA Analytics Dashboard
    await page.getByRole('link', { name: 'EDA Dashboard' }).first().click();
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.content()).resolves.toContain('Mental Health Risk Distribution');

    // 4. Navigate to ML Model Benchmarks
    await page.getByRole('link', { name: 'ML Benchmarks' }).first().click();
    await expect(page).toHaveURL(/\/models/);
    await expect(page.content()).resolves.toContain('C4.5 Decision Tree Classifier');

    // 5. Navigate to Assessment History Log
    await page.getByRole('link', { name: 'Assessment Log' }).first().click();
    await expect(page).toHaveURL(/\/history/);
    await expect(page.getByRole('button', { name: /Export CSV Report/i })).toBeVisible();

    // 6. Test System Health Endpoint
    const response = await page.request.get('/api/health');
    expect(response.status()).toBe(200);
    const healthJson = await response.json();
    expect(healthJson.status).toBe('healthy');
  });
});
