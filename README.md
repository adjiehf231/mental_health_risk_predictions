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

## 💻 Panduan Running di Lokal & Akses Jaringan Lokal (LAN / Wi-Fi)

### 📌 Prerequisites
Pastikan perangkat Anda sudah terinstal:
- **Node.js**: v18.0.0+ (`node -v`)
- **Python**: v3.10.0+ (`python --version`)
- **Git**: (`git --version`)

---

### 🌐 Cara Agar Aplikasi Dapat Diakes Perangkat Lain di Jaringan Wi-Fi / LAN yang Sama

Jika Anda ingin perangkat lain (HP, Tablet, Laptop kawan) mengakses aplikasi yang berjalan di laptop/komputer Anda dalam satu jaringan Wi-Fi:

#### Langkah 1: Cek Alamat IP Lokal Komputer Anda
Buka PowerShell / Command Prompt dan jalankan:
```cmd
ipconfig
```
Cari bagian **IPv4 Address** pada adapter Wi-Fi / Ethernet Anda. Contoh: `192.168.1.15` atau `192.168.100.25`.

#### Langkah 2: Jalankan Server Lokal dengan Bind Host `0.0.0.0`
Buka 2 terminal terpisah:

**Terminal 1 (Python Serverless ML Engine on Port 5328)**:
```bash
python -m uvicorn api.py.index:app --host 0.0.0.0 --port 5328 --reload
```

**Terminal 2 (Next.js Dev Server on Port 3000)**:
```bash
npm run dev:host
```
*Perintah `npm run dev:host` akan mengikat Next.js ke host `0.0.0.0:3000` sehingga dapat menerima koneksi dari jaringan luar.*

#### Langkah 3: Akses dari Perangkat Lain
Perangkat lain yang terhubung ke Wi-Fi yang sama cukup membuka browser dan mengetik alamat:
```text
http://<IP_KOMPUTER_ANDA>:3000
```
*Contoh: `http://192.168.1.15:3000` atau `http://192.168.100.25:3000`.*

---

### 🟢 Opsi Standar Running Lokal (Personal Localhost Only)

```bash
# Install dependencies
npm install
pip install -r requirements.txt
pip install -r api/py/requirements.txt

# Terminal 1 (Python API)
python -m uvicorn api.py.index:app --port 5328 --reload

# Terminal 2 (Next.js)
npm run dev
```
Akses di browser laptop Anda: **`http://localhost:3000`**.

---

### 5️⃣ Perintah Test & Verifikasi Rilis
```bash
# Run Pre-flight Release Verification Tool
npm run preflight

# Run Vitest Unit Tests
npm run test

# Build Production Next.js Bundle
npm run build

# Run Playwright E2E Cross-browser Tests
npm run test:e2e
```
