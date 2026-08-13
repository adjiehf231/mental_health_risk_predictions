# QA Automation & Quality Assurance Manual 🧪
## Mental Health Risk Prediction & Assessment Platform

> **Target Platform**: Next.js 14 App Router, Python Serverless API, Supabase PostgreSQL
> **Author**: Adjie Hari Fajar, S.Kom
> **Test Status**: ✅ 100% Passed (Zero Build/Type Errors)

---

## 📐 1. Quality Assurance Architecture

```mermaid
graph TD
    Preflight[Pre-flight Verification Script] --> ManifestCheck[Core File Manifest Check]
    Preflight --> VitestSuite[Vitest Unit Test Suite]
    Preflight --> BuildCheck[Next.js Production Build Validation]
    
    VitestSuite --> TypeTests[lib/__tests__/types.test.ts]
    VitestSuite --> SupabaseTests[lib/__tests__/supabase.test.ts]
    VitestSuite --> HealthTests[lib/__tests__/health.test.ts]
    
    E2ESuite[Playwright Cross-Browser E2E] --> NavE2E[e2e/navigation.spec.ts]
    E2ESuite --> PredictE2E[e2e/prediction.spec.ts]
    E2ESuite --> ReleaseE2E[e2e/release-readiness.spec.ts]
```

---

## 🧪 2. Automated Test Commands

```bash
# 1. Run Pre-flight Production Verification Tool
npm run preflight

# 2. Run Vitest Unit Tests
npm run test

# 3. Validate Next.js Production Build Compilation
npm run build

# 4. Run Playwright E2E Tests (Chromium, Firefox, WebKit)
npm run test:e2e

# 5. Run Playwright Interactive UI Test Runner
npm run test:e2e:ui
```

---

## 🔒 3. Security & HTTPS Audit Suite

- **HSTS Enforcement**: `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- **Frame Protection**: `X-Frame-Options: DENY`
- **MIME Protection**: `X-Content-Type-Options: nosniff`
- **Permissions Policy**: `camera=(), microphone=(), geolocation=()`
- **Health Diagnostic Endpoint**: `GET /api/health` returning 200 OK status.
