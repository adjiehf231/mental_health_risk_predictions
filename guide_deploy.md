# Vercel & Supabase Complete Deployment & Local Execution Guide 🚀
## Mental Health Risk Prediction & Assessment Platform

> **Cross References**: [README.md](README.md) \| [PRD.md](PRD.md) \| [qa_automation.md](qa_automation.md) \| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)

This guide provides step-by-step instructions for running the **Mental Health Risk Prediction** platform locally and deploying it to **Vercel** (Frontend & Python Serverless ML Engine) and **Supabase** (PostgreSQL Database & Authentication).

---

## 🏗️ Architecture & Topology

```mermaid
graph TD
    Client([Global Users / Browsers]) <-->|HTTPS Edge CDN| Vercel[Vercel Global Edge Network]
    subgraph Vercel Infrastructure
        Vercel <--> NextJS[Next.js 14/15 App Router Frontend]
        NextJS <-->|Rewrite /api/py/predict| PyServerless[Python 3.10 Serverless Runtime]
        PyServerless <-->|Load Models| MLModels[scikit-learn DecisionTree joblib]
    end
    subgraph Supabase Infrastructure
        NextJS <-->|@supabase/supabase-js| SupabaseDB[(Supabase PostgreSQL Database)]
        SupabaseDB <--> RLS[Row Level Security & Indexes]
    end
```

---

## 💻 Panduan Lengkap Running di Lokal (Local Setup & Run Guide)

### 📌 Prerequisites
- **Node.js**: v18.0.0+ (`node -v`)
- **Python**: v3.10.0+ (`python --version`)
- **Git**: (`git --version`)

---

### 1️⃣ Clone & Directory Setup
```bash
git clone https://github.com/adjiehf231/mental_health_risk_predictions.git
cd mental_health_risk_predictions
```

---

### 2️⃣ Install Dependencies
```bash
# Node.js dependencies
npm install

# Python ML dependencies
pip install -r requirements.txt
pip install -r api/py/requirements.txt
```

---

### 3️⃣ Setup Environment Variables (`.env.local`)
Create `.env.local` in project root:
```bash
cp .env.example .env.local
```
Add your Supabase credentials (optional: if omitted, app operates automatically in **Local Demo Mode**):
```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project-ref.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

---

### 4️⃣ Running Local Servers

#### 🟢 Option A: Fullstack Local Execution (Frontend Next.js + Python ML Server API)
Use 2 separate terminal windows:

**Terminal 1 (Python Serverless API Endpoint)**:
```bash
python -m uvicorn api.py.index:app --port 5328 --reload
```
*Health check URL: `http://127.0.0.1:5328/api/py/health`.*

**Terminal 2 (Next.js App Server)**:
```bash
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser. Next.js automatically rewrites requests from `/api/py/*` to `http://127.0.0.1:5328`.

---

#### 🔵 Option B: Standalone Next.js Mode
If running without python uvicorn:
```bash
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)**. The app automatically uses client fallback estimation for instant predictions.

---

### 5️⃣ Local Testing Commands
```bash
# Pre-flight Release Verification Tool
npm run preflight

# Vitest Unit Tests
npm run test

# Next.js Production Build Test
npm run build

# Playwright E2E Cross-browser Tests
npm run test:e2e
```

---

## 🗄️ Step 1: Supabase Database Setup

### 1.1 Create a New Supabase Project
1. Log in to [Supabase Dashboard](https://supabase.com/dashboard).
2. Click **New Project**.
3. Select your organization and enter:
   - **Name**: `mental-health-risk-predictions`
   - **Database Password**: Generates a strong password.
   - **Region**: Choose region closest to your users.
4. Click **Create new project**.

### 1.2 Run SQL Schema Initialization
1. In Supabase sidebar, click **SQL Editor**.
2. Paste the contents of [`supabase_schema.sql`](supabase_schema.sql) and click **Run**:

```sql
CREATE TABLE IF NOT EXISTS public.assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    age INT NOT NULL,
    gender TEXT NOT NULL,
    marital_status TEXT NOT NULL,
    education_level TEXT NOT NULL,
    employment_status TEXT NOT NULL,
    sleep_hours NUMERIC(4,1) NOT NULL,
    physical_activity_hours_per_week NUMERIC(4,1) DEFAULT 0,
    screen_time_hours_per_day NUMERIC(4,1) DEFAULT 0,
    social_support_score INT DEFAULT 5,
    work_stress_level INT NOT NULL,
    job_satisfaction_score INT DEFAULT 5,
    financial_stress_level INT DEFAULT 5,
    anxiety_score INT NOT NULL,
    depression_score INT NOT NULL,
    panic_attack_history INT DEFAULT 0,
    family_history_mental_illness INT DEFAULT 0,
    substance_use INT DEFAULT 0,
    risk_level TEXT NOT NULL,
    confidence NUMERIC(5,2) NOT NULL,
    probabilities JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.assessments ENABLE ROW LEVEL SECURITY;

-- Create Policies for Public Access
CREATE POLICY "Allow public read access" ON public.assessments FOR SELECT USING (true);
CREATE POLICY "Allow public insert access" ON public.assessments FOR INSERT WITH CHECK (true);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_assessments_risk_level ON public.assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON public.assessments(created_at DESC);
```

### 1.3 Copy Credentials
Go to **Project Settings** -> **API** and copy `Project URL` and `anon public key`.

---

## ⚡ Step 2: Vercel Deployment Setup

1. Push your changes to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete Next.js app with Python serverless API and Supabase integration"
   git push origin main
   ```
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) -> **Add New...** -> **Project**.
3. Import `adjiehf231/mental_health_risk_predictions`.
4. Add Environment Variables:
   - `NEXT_PUBLIC_SUPABASE_URL` = `https://<your-project-ref>.supabase.co`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = `<your-anon-key>`
5. Click **Deploy**. Vercel will build the frontend pages and deploy the Python serverless ML endpoint!

---

## 🔍 Step 3: Post-Deployment Verification

Verify live serverless prediction endpoint with `curl`:
```bash
curl -X POST https://mental-health-risk-predictions.vercel.app/api/py/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "Female",
    "marital_status": "Single",
    "education_level": "Bachelor",
    "employment_status": "Employed",
    "sleep_hours": 7.0,
    "work_stress_level": 5,
    "anxiety_score": 4,
    "depression_score": 3
  }'
```

---

## 🛠️ Troubleshooting & Common Fixes

- **Model File Not Found**: Ensure `models/*.pkl` files are committed to git.
- **Supabase Disconnected**: Ensure `NEXT_PUBLIC_SUPABASE_URL` environment variables are added in Vercel settings.
- **Python Timeout**: Keep `api/py/requirements.txt` lightweight.
