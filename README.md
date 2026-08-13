# Mental Health Risk Prediction & Assessment Platform 🧠

[![Next.js](https://img.shields.io/badge/Next.js-14%2F15-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38BDF8?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase)](https://supabase.com/)
[![Vercel](https://img.shields.io/badge/Vercel-Deployment-000000?style=for-the-badge&logo=vercel)](https://vercel.com/)
[![Playwright](https://img.shields.io/badge/Playwright-QA_Automation-2EAD33?style=for-the-badge&logo=playwright)](https://playwright.dev/)

> Production-ready fullstack web application for predicting mental health risk levels (Low, Moderate, High) using Machine Learning (99.5% Decision Tree Accuracy). Built with Next.js 14/15, Tailwind CSS (Glassmorphism design system), Vercel Python Serverless ML Engine, and Supabase PostgreSQL.

---

## 📚 Complete Project Documentation

- 📄 **[PRD.md](PRD.md)** - Product Requirement Document outlining vision, user personas, sprint roadmap, technical specs, and feature requirements.
- 🚀 **[guide_deploy.md](guide_deploy.md)** - Step-by-Step Vercel & Supabase Deployment Manual, local execution guide, database setup, environment configuration, and verification.
- 🧪 **[qa_automation.md](qa_automation.md)** - QA Automation Manual for Playwright E2E tests, API testing suites, Vitest component tests, and CI/CD GitHub Actions.
- 📋 **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)** - Production Release Audit Checklist & pre-flight verification script guide.

---

## 💻 Panduan Lengkap Running di Lokal (Local Setup & Run Guide)

### 📌 Prerequisites
Pastikan perangkat Anda sudah terinstal:
- **Node.js**: v18.0.0 atau versi lebih baru (`node -v`)
- **Python**: v3.10.0 atau versi lebih baru (`python --version`)
- **Git**: (`git --version`)

---

### 1️⃣ Clone Repository & Setup Directory
```bash
git clone https://github.com/adjiehf231/mental_health_risk_predictions.git
cd mental_health_risk_predictions
```

---

### 2️⃣ Install Dependencies (Node.js & Python)
```bash
# Install dependencies frontend (Next.js, React, Tailwind, Supabase, Recharts)
npm install

# Install dependencies Python ML engine
pip install -r requirements.txt
pip install -r api/py/requirements.txt
```

---

### 3️⃣ Konfigurasi Environment Variables (`.env.local`)
Buat file `.env.local` di direktori utama (root):
```bash
cp .env.example .env.local
```
Isi file `.env.local` dengan kredensial Supabase Anda (Opsional: Jika tidak diisi, aplikasi secara otomatis berjalan dalam **Demo Mode** dengan penyimpanan lokal):
```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project-ref.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

---

### 4️⃣ Cara Menjalankan Aplikasi di Lokal

#### 🟢 Opsi A: Fullstack Mode (Frontend Next.js + Python ML Serverless API Local)
Gunakan 2 terminal terpisah:

**Terminal 1 (Python ML API Endpoint on Port 5328)**:
```bash
python -m uvicorn api.py.index:app --port 5328 --reload
```
*API health check akan aktif di `http://127.0.0.1:5328/api/py/health`.*

**Terminal 2 (Next.js Web Server on Port 3000)**:
```bash
npm run dev
```
Buka browser dan akses **[http://localhost:3000](http://localhost:3000)**. Next.js secara otomatis meretautkan rute `/api/py/*` ke server Python lokal.

---

#### 🔵 Opsi B: Standalone Next.js Mode (Client Fallback Engine)
Jika Anda hanya ingin menjalankan server Next.js (tanpa Python uvicorn):
```bash
npm run dev
```
Akses **[http://localhost:3000](http://localhost:3000)**. Sistem secara otomatis menggunakan client fallback engine untuk prediksi instant.

---

### 5️⃣ Menjalankan Test Automated & Verifikasi Rilis
```bash
# Run Pre-flight Release Verification Tool
npm run preflight

# Run Vitest Unit Tests
npm run test

# Build Production Next.js Bundle
npm run build

# Run Playwright E2E Cross-browser Tests (Chromium, Firefox, WebKit)
npm run test:e2e

# Run Playwright Interactive UI Mode
npm run test:e2e:ui
```

---

## 🗓️ Development Phases & Agile Sprint Roadmap

```
Sprint 1: Architecture Foundation & Database Setup (Days 1–4)
  ├── Phase 1: Next.js 14/15 App Router & Glassmorphism Design System Setup
  └── Phase 2: Supabase PostgreSQL Schema (supabase_schema.sql) & Client Integration

Sprint 2: Modern UI/UX & Interactive Frontend Pages (Days 5–8)
  └── Phase 3: Landing Page, AI Risk Predictor, EDA Dashboard, ML Models, & History

Sprint 3: Serverless ML Inference Engine (Days 9–11)
  └── Phase 4: Vercel Python Serverless ML API Endpoint (/api/py/predict)

Sprint 4: QA Automation & Cross-Browser Testing (Days 12–14)
  └── Phase 5: Playwright E2E Suites, API Verification, & GitHub Actions CI

Sprint 5: Production Launch & Deployment (Days 15–16)
  └── Phase 6: Production Supabase Sync & Vercel Deployment Release
```

---

## ✨ Key Features

- **🔮 Interactive AI Risk Predictor**: Real-time evaluation of patient profiles across 15 key demographic, lifestyle, and psychological features.
- **📄 Printable Clinical Summary Report**: One-click printable diagnostic summary modal (`window.print()`) with patient profile parameter breakdowns.
- **📊 Comprehensive EDA Dashboard**: Visual exploration of 25,000+ dataset records, correlation heatmaps, age vs. stress distributions, and demographics.
- **🤖 Machine Learning Benchmarks**: Comparison of 5 algorithms (Decision Tree 99.5%, Random Forest 97.2%, SVM 93.5%, KNN 91.8%, Naive Bayes 89.4%) with feature importance analysis.
- **📜 Supabase Assessment History**: Real-time persistence of risk evaluations, filtering, search, and CSV data exports.
- **🎨 Glassmorphism & Modern UI/UX**: Dark/Light mode theme support, responsive 1-5 column grid layouts, interactive Recharts graphs, and smooth micro-animations.
- **⚡ Vercel Python Serverless API**: Direct execution of scikit-learn models natively inside Vercel serverless environment.

---

Made with ❤️ for Mental Health Awareness & Healthcare Technology.
