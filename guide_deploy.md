# Panduan Lengkap & Detail Step-by-Step Deployment: Vercel & Supabase 🚀
## Mental Health Risk Prediction & Assessment Platform

> **Target Platform**: Vercel (Next.js 14 Frontend + Python 3.10 Serverless ML Engine) & Supabase (PostgreSQL Database + Auth)
> **Owner**: Adjie Hari Fajar, S.Kom
> **Versi Dokumen**: 2.0.0 (Lengkap & Terverifikasi Produksi)

---

## 🏗️ 1. Topology & Pre-Deployment Checklist

### 📐 Skema Arsitektur Deployment
```mermaid
graph TD
    User([Pengguna Global / Browser]) <-->|HTTPS HSTS Edge| Vercel[Vercel Global Edge CDN]
    subgraph Vercel Infrastructure
        Vercel <--> NextJS[Next.js 14/15 App Router Frontend]
        NextJS <-->|Rewrite /api/py/predict| PyServerless[Python 3.10 Serverless Runtime]
        PyServerless <-->|Load Joblib Models| MLModels[scikit-learn Decision Tree .pkl]
    end
    subgraph Supabase Infrastructure
        NextJS <-->|@supabase/supabase-js| SupabaseDB[(Supabase PostgreSQL Database)]
        SupabaseDB <--> RLS[Row Level Security & Index Tabel]
    end
```

### 📋 Checklist Prasyarat & Kredensial
Sebelum memulai deployment, pastikan Anda telah memiliki:
1. Akun **[GitHub](https://github.com)** dengan repository terhubung: `https://github.com/adjiehf231/mental_health_risk_predictions`.
2. Akun **[Supabase](https://supabase.com)** (Gratis / Pro).
3. Akun **[Vercel](https://vercel.com)** (Gratis / Pro).

---

## 🗄️ 2. Langkah Demi Langkah Setup Database Supabase (Detail)

### 📍 Langkah 2.1: Buat Project Baru di Supabase
1. Buka browser dan login ke **[Supabase Dashboard](https://supabase.com/dashboard)**.
2. Klik tombol hijau **New Project**.
3. Pilih **Organization** Anda.
4. Isi formulir konfigurasi project:
   - **Name**: `mental-health-risk-predictions`
   - **Database Password**: Masukkan kata sandi kuat (simpan kata sandi ini).
   - **Region**: Pilih lokasi terdekat (misal: `Singapore` atau `Tokyo`).
   - **Pricing Plan**: Pilih `Free` tier.
5. Klik **Create new project**. Tunggu 1–2 menit hingga database selesai disiapkan oleh Supabase.

---

### 📍 Langkah 2.2: Eksekusi Skrip SQL Database Schema & RLS
1. Di sidebar sebelah kiri Supabase Dashboard, klik ikon **SQL Editor** (ikon `<>`).
2. Klik tombol **+ New query**.
3. Salin dan tempel (copy-paste) seluruh kode SQL di bawah ini:

```sql
-- 1. Buat Tabel Penilaian Risiko Kesehatan Mental
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

-- 2. Aktifkan Row Level Security (RLS)
ALTER TABLE public.assessments ENABLE ROW LEVEL SECURITY;

-- 3. Buat Kebijakan Akses Publik (Read & Insert)
CREATE POLICY "Allow public read access" 
ON public.assessments FOR SELECT 
USING (true);

CREATE POLICY "Allow public insert access" 
ON public.assessments FOR INSERT 
WITH CHECK (true);

-- 4. Buat Index Performa Query
CREATE INDEX IF NOT EXISTS idx_assessments_risk_level ON public.assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON public.assessments(created_at DESC);
```

4. Klik tombol **Run** (atau tekan `Ctrl + Enter`). Pastikan muncul pesan sukses: `Success. No rows returned`.

---

### 📍 Langkah 2.3: Ambil Kredensial API Supabase
1. Di sidebar Supabase, navigasi ke **Project Settings** ⚙️ ➔ **API**.
2. Salin nilai dari dua kredensial berikut:
   - **Project URL**: `https://<project-ref>.supabase.co`
   - **Project API keys (anon / public)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6...`

---

## ⚡ 3. Langkah Demi Langkah Deployment ke Vercel (Detail)

### 📍 Langkah 3.1: Hubungkan Repository GitHub ke Vercel
1. Login ke **[Vercel Dashboard](https://vercel.com/dashboard)**.
2. Klik tombol **Add New...** di pojok kanan atas ➔ pilih **Project**.
3. Di bagian **Import Git Repository**, cari repository GitHub Anda: `adjiehf231/mental_health_risk_predictions`.
4. Klik tombol **Import**.

---

### 📍 Langkah 3.2: Konfigurasi Build & Environment Variables
1. Pada halaman **Configure Project**:
   - **Project Name**: `mental-health-risk-predictions`
   - **Framework Preset**: `Next.js` (Otomatis terdeteksi).
   - **Root Directory**: `./` (Default).
2. Buka bagian **Environment Variables** dan tambahkan 2 variabel lingkungan berikut:

| Key Variable | Value / Isi | Deskripsi |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_SUPABASE_URL` | `https://<project-ref>.supabase.co` | Project URL dari Supabase (Langkah 2.3) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJhbGciOiJIUzI1NiIsInR5...` | Anon public key dari Supabase (Langkah 2.3) |

3. Klik tombol **Add** untuk setiap variabel.

---

### 📍 Langkah 3.3: Eksekusi Deploy & Verifikasi Build Logs
1. Klik tombol **Deploy**.
2. Vercel akan secara otomatis:
   - Mengompilasi halaman statis & dinamis Next.js 14.
   - Menyiapkan runtime **Python 3.10 Serverless Engine** untuk rute `/api/py/predict`.
   - Menerapkan header keamanan HSTS HTTPS (`Strict-Transport-Security`).
3. Tunggu hingga muncul ucapan selamat: **Congratulations! Your project is live on Vercel.**

---

## 🧪 4. Verifikasi Pasca-Deployment (Post-Deployment Testing)

### 🔍 1. Uji Endpoint Diagnostics Health API
Buka terminal dan jalankan perintah curl:
```bash
curl -i https://mental-health-risk-predictions.vercel.app/api/health
```
*Respon harus mengembalikan HTTP status `200 OK` dengan payload JSON `{"status": "healthy"}`.*

### 🔍 2. Uji Prediksi Python Serverless ML Endpoint
Jalankan tes prediksi menggunakan payload JSON pasien:
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
*Respon harus mengembalikan hasil prediksi klasifikasi risiko dan probabilitas dari model Decision Tree.*

---

## 🌐 5. Panduan Akses Jaringan Lokal (LAN / Wi-Fi Sharing)

Jika Anda ingin menjalankan aplikasi di komputer Anda dan dapat dibuka oleh HP / laptop lain di jaringan Wi-Fi yang sama:

1. **Cek Alamat IP Komputer Anda**:
   ```cmd
   ipconfig
   ```
   *Catat alamat IPv4 Anda, contoh: `192.168.1.15`.*

2. **Jalankan Server Lokal dengan Skrip Runner**:
   - Terminal 1 (Python API): `python -m uvicorn api.py.index:app --host 0.0.0.0 --port 5328 --reload`
   - Terminal 2 (Next.js LAN): `npm run dev:host`

3. **Akses dari Perangkat Lain**:
   Buka browser di HP/laptop lain yang terhubung ke Wi-Fi yang sama dan buka:
   ```text
   http://<IP_KOMPUTER_ANDA>:3000
   ```

---

## 🛠️ 6. Troubleshooting & Solusi Kendala Umum

### 🔴 Error: `ERR_ADDRESS_INVALID` saat buka `http://0.0.0.0:3000`
- **Sebab**: `0.0.0.0` adalah meta-address internal server, bukan URL IP valid di browser.
- **Solusi**: Gunakan **`http://localhost:3000`** di komputer server, atau **`http://192.168.x.x:3000`** di HP.

### 🔴 Error: `EADDRINUSE: address already in use :::3000`
- **Sebab**: Port 3000 sedang digunakan oleh proses Next.js lain.
- **Solusi**: Jalankan perintah `npx kill-port 3000` di PowerShell atau tutup terminal Next.js sebelumnya.

### 🔴 Error: `Supabase Disconnected / Local Demo Mode`
- **Sebab**: Variabel `NEXT_PUBLIC_SUPABASE_URL` belum ditambahkan di Vercel Environment Variables.
- **Solusi**: Masuk ke Vercel Settings ➔ Environment Variables ➔ tambahkan kredensial Supabase ➔ Lakukan **Redeploy**.
