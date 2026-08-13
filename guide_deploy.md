# Vercel & Supabase Complete Deployment Guide 🚀
## Mental Health Risk Prediction & Assessment Platform

> **Cross References**: [README.md](README.md) \| [PRD.md](PRD.md) \| [qa_automation.md](qa_automation.md)

This guide provides step-by-step instructions for deploying the **Mental Health Risk Prediction** web platform using **Vercel** (Frontend & Python Serverless ML Engine) and **Supabase** (PostgreSQL Database & Authentication).

---

## 🏗️ Architecture & Deployment Topology

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

## 📌 Prerequisites

Before beginning deployment, ensure you have:
- A **GitHub Account** (to host your repository).
- A **Vercel Account** (connected to GitHub).
- A **Supabase Account** ([supabase.com](https://supabase.com)).
- **Node.js 18+** and **Python 3.10+** installed locally.

---

## 🗄️ Step 1: Supabase Database Setup

### 1.1 Create a New Supabase Project
1. Log in to [Supabase Dashboard](https://supabase.com/dashboard).
2. Click **New Project**.
3. Select your organization and enter:
   - **Name**: `mental-health-risk-predictions`
   - **Database Password**: Generates a strong password (save securely).
   - **Region**: Choose the region closest to your target users (e.g., *Singapore / Southeast Asia*).
4. Click **Create new project** and wait 1–2 minutes for database provisioning.

### 1.2 Run SQL Schema & Table Initialization
1. In the Supabase sidebar, click **SQL Editor**.
2. Click **New query** and paste the content from [`supabase_schema.sql`](supabase_schema.sql):

```sql
-- Create assessments table for storing patient assessments
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

3. Click **Run** to execute the script. Verify that the table `assessments` is created under **Table Editor**.

### 1.3 Obtain API Credentials
1. Navigate to **Project Settings** -> **API**.
2. Note down the following keys:
   - **Project URL**: `https://<your-project-ref>.supabase.co`
   - **API Key (anon public)**: `eyJhbGciOi...`

---

## ⚡ Step 2: Vercel Deployment Setup

### 2.1 Push Code to GitHub
Ensure all code and model artifacts are committed and pushed to GitHub:
```bash
git add .
git commit -m "feat: complete Next.js app with Python serverless API and Supabase integration"
git push origin main
```

### 2.2 Import Project to Vercel
1. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** -> **Project**.
3. Select your GitHub repository: `adjiehf231/mental_health_risk_predictions`.
4. Configure Project Settings:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: `./` (leave default)

### 2.3 Add Environment Variables in Vercel
Expand the **Environment Variables** section and add:

| Key | Value | Description |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_SUPABASE_URL` | `https://<your-project-ref>.supabase.co` | Supabase Project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbGciOi...` | Supabase Public Anon Key |

### 2.4 Verify Serverless Engine (`vercel.json`)
Ensure `vercel.json` exists in the repository root with the following configuration:
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    },
    {
      "src": "api/py/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/py/predict",
      "dest": "api/py/index.py"
    }
  ]
}
```

### 2.5 Deploy Project
1. Click **Deploy**.
2. Vercel will build the Next.js static and server pages, install Python packages from `api/py/requirements.txt`, and deploy the Python Serverless function.
3. Upon completion, Vercel will provide your deployment domain (e.g. `https://mental-health-risk-predictions.vercel.app`).

---

## 🔍 Step 3: Post-Deployment Verification & Smoke Testing

### 1. Test Web Application Pages
- **Homepage (`/`)**: Verify hero section and navigation links.
- **AI Risk Predictor (`/prediction`)**: Adjust sliders and submit a patient assessment. Confirm that the result card displays confidence score, risk badge, and probability breakdown.
- **EDA Dashboard (`/dashboard`)**: Verify interactive Recharts graphs render dataset distributions.
- **ML Models (`/models`)**: Confirm decision tree model metrics (99.5% accuracy) display properly.
- **Assessment History (`/history`)**: Check that newly submitted assessments appear in the Supabase history table.

### 2. Test Serverless ML API Endpoint
Using `curl` or Postman, verify the API endpoint directly:
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
**Expected Response**:
```json
{
  "prediction": 0,
  "risk_label": "Low Risk (0)",
  "confidence": 0.985,
  "probabilities": [0.985, 0.012, 0.003]
}
```

---

## 🌐 Step 4: Custom Domain & SSL (Optional)

1. In the Vercel Dashboard, go to **Settings** -> **Domains**.
2. Enter your custom domain (e.g. `mentalhealth.yourdomain.com`).
3. Follow Vercel's DNS prompt to add a `CNAME` or `A` record in your DNS provider (Cloudflare, Namecheap, GoDaddy).
4. Vercel will automatically issue a free SSL certificate.

---

## 🛠️ Troubleshooting & Common Fixes

### Issue 1: `selected_features.pkl missing` or Model File Not Found
- **Root Cause**: Model `.pkl` files were omitted by `.gitignore`.
- **Fix**: Ensure `models/best_model.pkl`, `models/scaler.pkl`, `models/selector.pkl`, `models/encoder.pkl`, and `models/selected_features.pkl` are tracked in git.

### Issue 2: Supabase Connection Offline / RLS Error
- **Root Cause**: `NEXT_PUBLIC_SUPABASE_URL` missing or RLS policy blocking inserts.
- **Fix**: Verify environment variables in Vercel settings and ensure SQL RLS policies in Step 1.2 were executed.

### Issue 3: Vercel Python Function Timeout
- **Root Cause**: Heavy cold starts or large package imports.
- **Fix**: Ensure `api/py/requirements.txt` is kept lightweight (`scikit-learn==1.4.0`, `joblib==1.3.1`, `numpy==1.26.0`, `pandas==2.2.0`).
