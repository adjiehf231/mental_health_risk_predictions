# Product Requirement Document (PRD) 📋
## Mental Health Risk Prediction & Assessment Platform

> **Version**: 2.0.0 (Vercel & Supabase Edition)
> **Owner**: Adjie Hari Fajar, S.Kom
> **Status**: Production Release Approved ✅

---

## 🎯 1. Executive Summary & Vision

The **Mental Health Risk Prediction Platform** is an intelligent fullstack web application designed to evaluate patient mental health risk levels (Low, Moderate, High) in real-time. Powered by a Decision Tree machine learning model trained on 25,000+ Kaggle clinical records (99.5% accuracy, ROC-AUC 0.998), the platform features interactive EDA dashboards, cross-validation algorithm benchmarks, parameter-specific medical guidance, and real-time record logging.

---

## 👥 2. User Personas & Target Audience

- **Individuals & Patients**: Seeking instant, confidential mental health risk evaluations and actionable lifestyle advice.
- **Clinical Counselors & Healthcare Professionals**: Utilizing empirical risk scores, ROC-AUC metrics, and printable clinical summary reports for patient consultations.
- **Data Scientists & QA Engineers**: Inspecting cross-validation model performance, feature importance rankings, and automated test coverage.

---

## ⚙️ 3. Technical Architecture & Stack Topology

- **Frontend**: Next.js 14/15 App Router, TypeScript 5.5, Tailwind CSS 3.4 (Glassmorphic Design System).
- **Theme & Multilingual State**: React Context (`lib/AppContext.tsx`) for Dark/Light mode and ID/EN language switching.
- **Machine Learning Engine**: Vercel Python 3.10 Serverless Runtime (`api/py/index.py`) loading scikit-learn models (`best_model.pkl`, `scaler.pkl`, `selector.pkl`, `encoder.pkl`).
- **Database**: Supabase PostgreSQL database (`public.assessments` table) with Row Level Security (RLS) policies.
- **Security & HTTPS**: HSTS (`Strict-Transport-Security`), CSP, X-Frame-Options (`DENY`), X-Content-Type-Options (`nosniff`).

---

## 🗓️ 4. Agile Sprint & Development Roadmap

```
Sprint 1: Architecture Foundation & Database Setup (Phases 1-2)
  ├── Glassmorphism Design System & Next.js App Router
  └── Supabase PostgreSQL Schema & RLS Policies

Sprint 2: Core Platform Modules & Analytics (Phase 3)
  ├── AI Risk Predictor (SelectKBest 15 Features)
  ├── EDA Dashboard (Pearson r, Age Trends, Lifestyle Impact)
  ├── ML Model Benchmarks (5 Algorithms & ROC-AUC Curves)
  └── Assessment History Log & Printable Summary Modal

Sprint 3: Serverless ML Engine & Security Hardening (Phase 4)
  ├── Python 3.10 Serverless FastAPI Endpoint (/api/py/predict)
  └── HSTS HTTPS Headers, CSP, & Health Endpoint (/api/health)

Sprint 4: QA Automation & Release Verification (Phases 5-6)
  ├── Vitest Unit Tests & Playwright E2E Suites
  └── Pre-flight Release Verification Script (npm run preflight)
```

---

## 🔒 5. Security & Compliance Standards

- **HTTPS Enforcement**: HSTS `max-age=63072000; includeSubDomains; preload`.
- **Row Level Security**: Public insert and select policies on Supabase `assessments` table.
- **Input Sanitization**: Zod parameter validation and scikit-learn feature encoding.

---

## 🧪 6. Quality Assurance & Test Verification

- **Vitest Unit Tests**: `npm run test` (100% pass rate).
- **Playwright E2E Suites**: `npm run test:e2e` (Chromium, Firefox, WebKit).
- **Pre-flight Script**: `npm run preflight` (Verifies manifests, unit tests, and production build).
