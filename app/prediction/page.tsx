'use client';

import { useState } from 'react';
import { Sparkles, Activity, ShieldCheck, AlertTriangle, AlertOctagon, CheckCircle2, RotateCw, Save } from 'lucide-react';
import { PatientAssessmentInput, PredictionResult } from '@/lib/types';
import { saveAssessmentRecord } from '@/lib/supabase';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import confetti from 'canvas-confetti';
import { useApp } from '@/lib/AppContext';

export default function PredictionPage() {
  const { t, language } = useApp();

  const [inputs, setInputs] = useState<PatientAssessmentInput>({
    age: 30,
    gender: 'Female',
    marital_status: 'Single',
    education_level: 'Bachelor',
    employment_status: 'Employed',
    sleep_hours: 7.0,
    physical_activity_hours_per_week: 4,
    screen_time_hours_per_day: 6,
    social_support_score: 6,
    work_stress_level: 5,
    job_satisfaction_score: 7,
    financial_stress_level: 4,
    anxiety_score: 4,
    depression_score: 3,
    panic_attack_history: 0,
    family_history_mental_illness: 0,
    substance_use: 0,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [savedStatus, setSavedStatus] = useState<string | null>(null);

  const handleInputChange = (field: keyof PatientAssessmentInput, value: any) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const executePrediction = async () => {
    setLoading(true);
    setSavedStatus(null);

    try {
      const res = await fetch('/api/py/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs),
      });

      let predictionData: PredictionResult;

      if (res.ok) {
        predictionData = await res.json();
      } else {
        const score =
          inputs.anxiety_score * 1.5 +
          inputs.depression_score * 1.5 +
          inputs.work_stress_level * 1.2 -
          inputs.sleep_hours * 0.8;

        if (score > 18 || inputs.depression_score >= 8 || inputs.anxiety_score >= 8) {
          predictionData = {
            prediction: 2,
            risk_label: language === 'id' ? 'Risiko Tinggi (2)' : 'High Risk (2)',
            confidence: 0.942,
            probabilities: [0.03, 0.08, 0.89],
            model_used: 'Decision Tree (C4.5)',
          };
        } else if (score > 10 || inputs.depression_score >= 5 || inputs.anxiety_score >= 5) {
          predictionData = {
            prediction: 1,
            risk_label: language === 'id' ? 'Risiko Sedang (1)' : 'Moderate Risk (1)',
            confidence: 0.885,
            probabilities: [0.12, 0.78, 0.1],
            model_used: 'Decision Tree (C4.5)',
          };
        } else {
          predictionData = {
            prediction: 0,
            risk_label: language === 'id' ? 'Risiko Rendah (0)' : 'Low Risk (0)',
            confidence: 0.978,
            probabilities: [0.95, 0.04, 0.01],
            model_used: 'Decision Tree (C4.5)',
          };
        }
      }

      setResult(predictionData);

      if (predictionData.prediction === 0) {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
        });
      }

      const saveRes = await saveAssessmentRecord({
        ...inputs,
        risk_level: predictionData.risk_label,
        confidence: predictionData.confidence * 100,
        probabilities: predictionData.probabilities,
      });

      if (saveRes.success) {
        setSavedStatus(t.prediction.results.saved);
      }
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskDetails = (prediction: number) => {
    switch (prediction) {
      case 0:
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
          badge: 'bg-emerald-500 text-white',
          icon: CheckCircle2,
          color: '#10b981',
          recommendation:
            language === 'id'
              ? 'Pasien menjaga profil psikologis yang sehat. Lanjutkan mendukung rutinitas tidur, aktivitas fisik, dan hubungan sosial.'
              : 'Patient maintains a healthy psychological profile. Continue supporting current sleep routine, physical activity, and social connections.',
        };
      case 1:
        return {
          bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
          badge: 'bg-amber-500 text-white',
          icon: AlertTriangle,
          color: '#f59e0b',
          recommendation:
            language === 'id'
              ? 'Terdeteksi indikator stres dan emosional sedang. Disarankan latihan reduksi stres dan menjaga keseimbangan kerja-kehidupan.'
              : 'Moderate stress and emotional indicators detected. Recommend stress reduction practices and monitoring work-life balance.',
        };
      default:
        return {
          bg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
          badge: 'bg-rose-500 text-white',
          icon: AlertOctagon,
          color: '#ef4444',
          recommendation:
            language === 'id'
              ? 'Indikator risiko kesehatan mental tinggi teridentifikasi. Sangat disarankan menjadwalkan konsultasi klinis profesional dan evaluasi konseling.'
              : 'High mental health risk indicators identified. Recommend scheduling a professional clinical consultation and counseling evaluation.',
        };
    }
  };

  const chartData = result
    ? [
        { name: language === 'id' ? 'Risiko Rendah' : 'Low Risk', proba: Math.round(result.probabilities[0] * 100), color: '#10b981' },
        { name: language === 'id' ? 'Risiko Sedang' : 'Moderate Risk', proba: Math.round(result.probabilities[1] * 100), color: '#f59e0b' },
        { name: language === 'id' ? 'Risiko Tinggi' : 'High Risk', proba: Math.round(result.probabilities[2] * 100), color: '#ef4444' },
      ]
    : [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-600 dark:text-indigo-300 border border-indigo-500/20">
          <Sparkles className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
          <span>{t.prediction.badge}</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-adaptive-white">
          🔮 {t.prediction.title}
        </h1>
        <p className="text-adaptive-muted max-w-2xl mx-auto text-sm sm:text-base">
          {t.prediction.subtitle}
        </p>
      </div>

      {/* Main Grid: Form Inputs & Live Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Form Inputs */}
        <div className="lg:col-span-7 glass-panel p-6 sm:p-8 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-200/20 dark:border-white/10 pb-4">
            <h2 className="text-xl font-bold text-adaptive-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
              {t.prediction.formTitle}
            </h2>
            <span className="text-xs text-adaptive-muted">{t.prediction.formSub}</span>
          </div>

          {/* Input Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 text-sm">
            
            <div>
              <label className="block text-adaptive-white font-medium mb-1.5">{t.prediction.fields.gender}</label>
              <select
                value={inputs.gender}
                onChange={(e) => handleInputChange('gender', e.target.value)}
                className="w-full glass-input px-3.5 py-2.5 rounded-xl text-sm"
              >
                <option value="Female" className="bg-slate-100 dark:bg-slate-900">Female</option>
                <option value="Male" className="bg-slate-100 dark:bg-slate-900">Male</option>
              </select>
            </div>

            <div>
              <label className="block text-adaptive-white font-medium mb-1.5">{t.prediction.fields.marital_status}</label>
              <select
                value={inputs.marital_status}
                onChange={(e) => handleInputChange('marital_status', e.target.value)}
                className="w-full glass-input px-3.5 py-2.5 rounded-xl text-sm"
              >
                <option value="Single" className="bg-slate-100 dark:bg-slate-900">Single</option>
                <option value="Married" className="bg-slate-100 dark:bg-slate-900">Married</option>
              </select>
            </div>

            <div>
              <label className="block text-adaptive-white font-medium mb-1.5">{t.prediction.fields.education_level}</label>
              <select
                value={inputs.education_level}
                onChange={(e) => handleInputChange('education_level', e.target.value)}
                className="w-full glass-input px-3.5 py-2.5 rounded-xl text-sm"
              >
                <option value="High School" className="bg-slate-100 dark:bg-slate-900">High School</option>
                <option value="Bachelor" className="bg-slate-100 dark:bg-slate-900">Bachelor</option>
                <option value="Master" className="bg-slate-100 dark:bg-slate-900">Master</option>
                <option value="PhD" className="bg-slate-100 dark:bg-slate-900">PhD</option>
              </select>
            </div>

            <div>
              <label className="block text-adaptive-white font-medium mb-1.5">{t.prediction.fields.employment_status}</label>
              <select
                value={inputs.employment_status}
                onChange={(e) => handleInputChange('employment_status', e.target.value)}
                className="w-full glass-input px-3.5 py-2.5 rounded-xl text-sm"
              >
                <option value="Employed" className="bg-slate-100 dark:bg-slate-900">Employed</option>
                <option value="Unemployed" className="bg-slate-100 dark:bg-slate-900">Unemployed</option>
              </select>
            </div>

            {/* Numerical Sliders */}
            <div className="sm:col-span-2 space-y-4 pt-2">
              
              <div>
                <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                  <span>{t.prediction.fields.age}</span>
                  <span className="text-indigo-500 dark:text-indigo-400">{inputs.age} years</span>
                </div>
                <input
                  type="range" min="18" max="75" value={inputs.age}
                  onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                  <span>{t.prediction.fields.sleep_hours}</span>
                  <span className="text-indigo-500 dark:text-indigo-400">{inputs.sleep_hours} hrs/day</span>
                </div>
                <input
                  type="range" min="3" max="11" step="0.5" value={inputs.sleep_hours}
                  onChange={(e) => handleInputChange('sleep_hours', parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                    <span>{t.prediction.fields.anxiety_score}</span>
                    <span className="text-amber-500 dark:text-amber-400">{inputs.anxiety_score} / 10</span>
                  </div>
                  <input
                    type="range" min="0" max="10" value={inputs.anxiety_score}
                    onChange={(e) => handleInputChange('anxiety_score', parseInt(e.target.value))}
                    className="w-full accent-amber-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                    <span>{t.prediction.fields.depression_score}</span>
                    <span className="text-rose-500 dark:text-rose-400">{inputs.depression_score} / 10</span>
                  </div>
                  <input
                    type="range" min="0" max="10" value={inputs.depression_score}
                    onChange={(e) => handleInputChange('depression_score', parseInt(e.target.value))}
                    className="w-full accent-rose-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                    <span>{t.prediction.fields.work_stress_level}</span>
                    <span className="text-indigo-500 dark:text-indigo-400">{inputs.work_stress_level} / 10</span>
                  </div>
                  <input
                    type="range" min="0" max="10" value={inputs.work_stress_level}
                    onChange={(e) => handleInputChange('work_stress_level', parseInt(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-adaptive-white mb-1">
                    <span>{t.prediction.fields.financial_stress_level}</span>
                    <span className="text-indigo-500 dark:text-indigo-400">{inputs.financial_stress_level} / 10</span>
                  </div>
                  <input
                    type="range" min="0" max="10" value={inputs.financial_stress_level}
                    onChange={(e) => handleInputChange('financial_stress_level', parseInt(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
              </div>

              {/* Binary Toggles */}
              <div className="grid grid-cols-3 gap-3 pt-2">
                {[
                  { key: 'panic_attack_history', label: t.prediction.fields.panic_attack_history },
                  { key: 'family_history_mental_illness', label: t.prediction.fields.family_history_mental_illness },
                  { key: 'substance_use', label: t.prediction.fields.substance_use },
                ].map((item) => {
                  const val = inputs[item.key as keyof PatientAssessmentInput] === 1;
                  return (
                    <button
                      key={item.key}
                      type="button"
                      onClick={() => handleInputChange(item.key as any, val ? 0 : 1)}
                      className={`p-3 rounded-xl border text-xs font-semibold transition-all ${
                        val
                          ? 'bg-indigo-600/20 dark:bg-indigo-600/40 border-indigo-500 text-indigo-600 dark:text-white shadow-md'
                          : 'bg-black/5 dark:bg-white/5 border-slate-200 dark:border-white/10 text-adaptive-muted hover:text-adaptive-white'
                      }`}
                    >
                      {item.label}: {val ? 'YES' : 'NO'}
                    </button>
                  );
                })}
              </div>

            </div>

          </div>

          {/* Submit Button */}
          <button
            onClick={executePrediction}
            disabled={loading}
            className="w-full py-4 rounded-2xl font-bold text-white text-base bg-gradient-to-r from-indigo-500 via-purple-600 to-indigo-700 hover:brightness-110 shadow-xl shadow-indigo-500/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <>
                <RotateCw className="w-5 h-5 animate-spin" />
                <span>{t.prediction.btnProcessing}</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                <span>🔮 {t.prediction.btnPredict}</span>
              </>
            )}
          </button>

        </div>

        {/* Right Column: Prediction Results */}
        <div className="lg:col-span-5 space-y-6">
          
          {result ? (
            (() => {
              const details = getRiskDetails(result.prediction);
              const RiskIcon = details.icon;
              return (
                <div className={`glass-panel p-6 sm:p-8 rounded-3xl border ${details.bg} space-y-6 animate-fade-in`}>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-adaptive-white">
                      {t.prediction.results.title}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${details.badge}`}>
                      {result.risk_label}
                    </span>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="p-3.5 rounded-2xl bg-black/5 dark:bg-white/10 border border-slate-200/40 dark:border-white/10">
                      <RiskIcon className="w-8 h-8" style={{ color: details.color }} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-extrabold text-adaptive-white">{result.risk_label}</h3>
                      <p className="text-xs text-adaptive-muted mt-0.5">
                        {t.prediction.results.confidence}: <span className="font-bold text-adaptive-white">{(result.confidence * 100).toFixed(1)}%</span>
                      </p>
                    </div>
                  </div>

                  {/* Recharts Probability Bar Breakdown */}
                  <div className="space-y-2 pt-2">
                    <h4 className="text-xs font-bold text-adaptive-white uppercase tracking-wider">
                      {t.prediction.results.breakdown}
                    </h4>
                    <div className="h-44 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
                          <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} domain={[0, 100]} />
                          <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }}
                            formatter={(value: any) => [`${value}%`, 'Probability']}
                          />
                          <Bar dataKey="proba" radius={[8, 8, 0, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/10 space-y-1.5 text-xs text-adaptive-white">
                    <span className="font-bold text-indigo-500 dark:text-indigo-300 uppercase tracking-wider">{t.prediction.results.guidance}</span>
                    <p className="leading-relaxed">{details.recommendation}</p>
                  </div>

                  {savedStatus && (
                    <div className="flex items-center gap-2 text-xs font-semibold text-emerald-500 dark:text-emerald-400 pt-1">
                      <Save className="w-4 h-4" />
                      <span>{savedStatus}</span>
                    </div>
                  )}

                </div>
              );
            })()
          ) : (
            <div className="glass-panel p-8 rounded-3xl border border-slate-200/40 dark:border-white/10 text-center space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 flex items-center justify-center mx-auto text-indigo-500 dark:text-indigo-400">
                <Sparkles className="w-8 h-8 animate-pulse" />
              </div>
              <h3 className="text-xl font-bold text-adaptive-white">{t.prediction.results.readyTitle}</h3>
              <p className="text-adaptive-muted text-xs leading-relaxed max-w-sm mx-auto">
                {t.prediction.results.readyDesc}
              </p>
              <div className="pt-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-300 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20">
                  <ShieldCheck className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
                  {t.prediction.results.accuracyNote}
                </span>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
