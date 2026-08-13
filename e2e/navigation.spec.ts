import { test, expect } from '@playwright/test';

test.describe('Platform Navigation E2E Suite', () => {
  test('Homepage loads correctly and displays title and CTA buttons', async ({ page }) => {
    await page.goto('/');
    
    // Check main heading
    await expect(page.locator('h1')).toContainText('Mental Health Risk');
    
    // Check navigation brand logo
    await expect(page.locator('header')).toContainText('MindRisk AI');
    
    // Check start assessment button
    const ctaButton = page.getByRole('link', { name: /Start Risk Assessment/i });
    await expect(ctaButton).toBeVisible();
  });

  test('Navbar links navigate to correct page routes', async ({ page }) => {
    await page.goto('/');

    // Click AI Predictor link
    await page.getByRole('link', { name: 'AI Predictor' }).first().click();
    await expect(page).toHaveURL(/\/prediction/);

    // Click EDA Dashboard link
    await page.getByRole('link', { name: 'EDA Dashboard' }).first().click();
    await expect(page).toHaveURL(/\/dashboard/);

    // Click ML Benchmarks link
    await page.getByRole('link', { name: 'ML Benchmarks' }).first().click();
    await expect(page).toHaveURL(/\/models/);

    // Click Assessment Log link
    await page.getByRole('link', { name: 'Assessment Log' }).first().click();
    await expect(page).toHaveURL(/\/history/);
  });
});
