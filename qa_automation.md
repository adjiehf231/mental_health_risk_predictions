# QA Automation Guide & Strategy 🧪
## Mental Health Risk Prediction Platform

> **Cross References**: [README.md](README.md) \| [PRD.md](PRD.md) \| [guide_deploy.md](guide_deploy.md)

This document outlines the QA Automation Strategy, test planning, setup instructions, and execution guides for the **Mental Health Risk Prediction** web platform.

---

## 1. QA Automation Architecture

```mermaid
graph TD
    A[GitHub Actions CI/CD Pipeline] --> B[Playwright E2E Suite]
    A --> C[Vitest Component & Unit Suite]
    A --> D[Serverless API Test Suite]
    B -->|Cross-browser Testing| E[Chromium / Firefox / WebKit]
    C -->|React & Lib Tests| F[Component & Utility Validation]
    D -->|HTTP Request Verification| G[/api/py/predict Endpoint]
```

---

## 2. Test Strategy & Scope

| Test Type | Framework | Coverage Scope | Target Command |
| :--- | :--- | :--- | :--- |
| **End-to-End (E2E)** | Playwright | Full user journey, slider inputs, prediction execution, history sync | `npm run test:e2e` |
| **API Testing** | Playwright / Vitest | Serverless Python ML API (`/api/py/predict`) latency & response verification | `npm run test:api` |
| **Unit & Component** | Vitest + React Testing Library | Component rendering, state updates, utility functions | `npm run test` |
| **Visual UI Tests** | Playwright Screenshots | Layout regression, responsive design verification | `npx playwright test --update-snapshots` |

---

## 3. Playwright E2E Test Specifications

### 3.1 Prediction Flow Test (`e2e/prediction.spec.ts`)
- **Objective**: Verify that adjusting sliders and clicking "Predict Risk" returns an accurate risk classification card and probability breakdown chart.
- **Steps**:
  1. Navigate to `/prediction`.
  2. Verify top 15 selected features sliders/dropdowns are rendered.
  3. Set Anxiety Score to `8`, Depression Score to `9`, Sleep Hours to `4`.
  4. Click `🔮 Predict Mental Health Risk`.
  5. Assert that the result card renders `High Risk (2)` or `Moderate Risk (1)`.
  6. Assert confidence badge and Recharts probability bar chart are visible.

### 3.2 Navigation & Routing Test (`e2e/navigation.spec.ts`)
- **Objective**: Verify top navbar links navigate correctly between app sections.
- **Steps**:
  1. Load homepage `/`.
  2. Click Navbar link `📊 Dashboard & EDA` -> verify URL is `/dashboard` and charts load.
  3. Click `🤖 ML Models` -> verify URL is `/models` and algorithm metrics table displays.
  4. Click `🔮 Prediction` -> verify URL is `/prediction`.
  5. Click `📜 History` -> verify URL is `/history`.

### 3.3 Serverless ML API Test (`e2e/api.spec.ts`)
- **Objective**: Direct HTTP API contract validation for `/api/py/predict`.
- **Payload Sample**:
```json
{
  "age": 32,
  "gender": "Female",
  "marital_status": "Single",
  "education_level": "Master",
  "employment_status": "Employed",
  "sleep_hours": 5.5,
  "physical_activity_hours_per_week": 2,
  "screen_time_hours_per_day": 8,
  "social_support_score": 4,
  "work_stress_level": 8,
  "job_satisfaction_score": 3,
  "financial_stress_level": 7,
  "anxiety_score": 8,
  "depression_score": 7,
  "panic_attack_history": 1,
  "family_history_mental_illness": 1,
  "substance_use": 0
}
```
- **Assertions**:
  - HTTP Status Code: `200 OK`.
  - JSON Body contains `prediction` (integer 0, 1, or 2), `risk_label` (string), `confidence` (float between 0 and 1), and `probabilities` (array of 3 floats summing to 1.0).

---

## 4. Setup & Running Tests Locally

### Prerequisites
- Node.js 18+ installed
- Python 3.10+ installed

### Step 1: Install Dependencies
```bash
npm install
npx playwright install --with-deps
```

### Step 2: Run Unit Tests
```bash
npm run test
```

### Step 3: Run Playwright E2E Tests
```bash
# Run headless
npm run test:e2e

# Run with interactive Playwright UI Mode
npx playwright test --ui

# Run single test file
npx playwright test e2e/prediction.spec.ts
```

---

## 5. CI/CD GitHub Actions Workflow Integration

Below is the automated workflow file `.github/workflows/qa_ci.yml`:

```yaml
name: QA Automation & Build Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    timeout-minutes: 15
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: 20
        cache: 'npm'

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: |
        npm ci
        pip install -r requirements.txt

    - name: Install Playwright Browsers
      run: npx playwright install --with-deps

    - name: Run Unit Tests
      run: npm run test

    - name: Build Next.js Application
      run: npm run build

    - name: Run Playwright E2E Tests
      run: npm run test:e2e

    - name: Upload Playwright Report
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: playwright-report
        path: playwright-report/
        retention-days: 14
```

---

## 6. Test Data & Edge Cases Covered

1. **Boundary Values**: Minimum and maximum allowed values for all numerical features (e.g., Sleep hours = 0 vs 24; Anxiety score = 0 vs 10).
2. **Missing Input Fallbacks**: Graceful validation errors if mandatory patient fields are missing.
3. **Database Disconnection Handling**: Frontend UI displays friendly notifications when Supabase connection is offline while still allowing local model predictions.
