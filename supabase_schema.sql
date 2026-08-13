-- Mental Health Risk Assessment Database Schema
-- Supabase PostgreSQL Script

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

-- Index optimizations for high-performance querying
CREATE INDEX IF NOT EXISTS idx_assessments_risk_level ON public.assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON public.assessments(created_at DESC);
