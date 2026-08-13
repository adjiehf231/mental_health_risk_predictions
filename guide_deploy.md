# Vercel & Supabase Complete Deployment & Local Network Guide 🚀
## Mental Health Risk Prediction & Assessment Platform

> **Cross References**: [README.md](README.md) \| [PRD.md](PRD.md) \| [qa_automation.md](qa_automation.md) \| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)

This guide provides step-by-step instructions for running the **Mental Health Risk Prediction** platform locally, sharing it over a local Wi-Fi / LAN network, and deploying it to **Vercel** and **Supabase**.

---

## 🌐 Local Network Sharing Guide (Akses via Wi-Fi / LAN)

To allow other devices (smartphones, laptops, tablets) on the same local Wi-Fi network to access your local development instance:

### Step 1: Find your Local IP Address
On Windows PowerShell / CMD:
```cmd
ipconfig
```
Note your **IPv4 Address** (e.g., `192.168.1.15`).

### Step 2: Start Local Host Binding
Run 2 separate terminal windows:

**Terminal 1 (Python ML API on 0.0.0.0:5328)**:
```bash
python -m uvicorn api.py.index:app --host 0.0.0.0 --port 5328 --reload
```

**Terminal 2 (Next.js Dev Server on 0.0.0.0:3000)**:
```bash
npm run dev:host
```

### Step 3: Connect from Other Devices
Other devices connected to the same Wi-Fi can open their browser and visit:
```text
http://<YOUR_IP_ADDRESS>:3000
```
*(Example: `http://192.168.1.15:3000`)*

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
Execute `supabase_schema.sql` in Supabase SQL Editor.

---

## ⚡ Step 2: Vercel Deployment Setup

1. Push changes to GitHub repository `adjiehf231/mental_health_risk_predictions`.
2. Connect Vercel to repository and add `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
3. Click **Deploy**.
