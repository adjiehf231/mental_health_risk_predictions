'use client';

import { X, Printer, Brain, ShieldCheck, Calendar, Activity, CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';
import { AssessmentRecord } from '@/lib/types';
import { useApp } from '@/lib/AppContext';

interface ClinicalReportModalProps {
  record: AssessmentRecord;
  onClose: () => void;
}

export default function ClinicalReportModal({ record, onClose }: ClinicalReportModalProps) {
  const { t, language } = useApp();

  const handlePrint = () => {
    window.print();
  };

  const getRiskColor = (risk: string) => {
    if (risk.includes('Low') || risk.includes('Rendah')) return { color: '#10b981', icon: CheckCircle2 };
    if (risk.includes('Moderate') || risk.includes('Sedang')) return { color: '#f59e0b', icon: AlertTriangle };
    return { color: '#ef4444', icon: AlertOctagon };
  };

  const riskInfo = getRiskColor(record.risk_level);
  const RiskIcon = riskInfo.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      
      {/* Modal Container */}
      <div className="relative w-full max-w-2xl glass-panel p-6 sm:p-8 rounded-3xl border border-slate-200/40 dark:border-white/20 shadow-2xl max-h-[90vh] overflow-y-auto space-y-6">
        
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-slate-200/20 dark:border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-adaptive-white">{t.report.title}</h2>
              <p className="text-xs text-adaptive-muted">{t.report.sub}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="p-2 rounded-xl bg-slate-200/60 dark:bg-white/10 text-slate-800 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-white/20 transition-colors"
              title={t.report.printBtn}
            >
              <Printer className="w-5 h-5" />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-xl bg-slate-200/60 dark:bg-white/10 text-slate-800 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-white/20 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Assessment Classification Banner */}
        <div className="p-5 rounded-2xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/10 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <RiskIcon className="w-8 h-8" style={{ color: riskInfo.color }} />
            <div>
              <span className="text-xs font-semibold text-adaptive-muted uppercase tracking-wider">{t.report.status}</span>
              <h3 className="text-2xl font-extrabold text-adaptive-white">{record.risk_level}</h3>
            </div>
          </div>

          <div className="text-right">
            <span className="text-xs text-adaptive-muted font-medium block">{t.report.conf}</span>
            <span className="text-2xl font-extrabold text-indigo-500 dark:text-indigo-400">{record.confidence.toFixed(1)}%</span>
          </div>
        </div>

        {/* Patient Profile Details Grid */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold text-adaptive-white uppercase tracking-wider flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
            {t.report.inputs}
          </h4>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Age / Gender</span>
              <span className="font-bold text-adaptive-white">{record.age} yrs • {record.gender}</span>
            </div>

            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Education Level</span>
              <span className="font-bold text-adaptive-white">{record.education_level}</span>
            </div>

            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Employment Status</span>
              <span className="font-bold text-adaptive-white">{record.employment_status}</span>
            </div>

            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Sleep Duration</span>
              <span className="font-bold text-adaptive-white">{record.sleep_hours} hrs/day</span>
            </div>

            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Anxiety / Depression</span>
              <span className="font-bold text-adaptive-white">{record.anxiety_score}/10 • {record.depression_score}/10</span>
            </div>

            <div className="p-3 rounded-xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5">
              <span className="text-adaptive-muted block">Work Stress Level</span>
              <span className="font-bold text-adaptive-white">{record.work_stress_level}/10</span>
            </div>
          </div>
        </div>

        {/* Clinical Guidance Section */}
        <div className="p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-adaptive-white space-y-1">
          <span className="font-bold text-indigo-600 dark:text-indigo-300 uppercase tracking-wider">{t.report.guidance}</span>
          <p className="leading-relaxed">
            {record.risk_level.includes('Low') || record.risk_level.includes('Rendah')
              ? language === 'id'
                ? 'Pasien menjaga profil psikologis yang sehat. Lanjutkan mendukung rutinitas tidur, aktivitas fisik, dan hubungan sosial.'
                : 'Patient maintains a healthy psychological profile. Continue supporting current sleep routine, physical activity, and social connections.'
              : record.risk_level.includes('Moderate') || record.risk_level.includes('Sedang')
              ? language === 'id'
                ? 'Terdeteksi indikator stres sedang. Disarankan latihan reduksi stres dan menjaga keseimbangan kerja-kehidupan.'
                : 'Moderate stress levels identified. Suggest stress reduction interventions and lifestyle consultation.'
              : language === 'id'
              ? 'Indikator risiko tinggi teridentifikasi. Sangat disarankan menjadwalkan konsultasi klinis profesional dan evaluasi konseling.'
              : 'High risk indicators present. Recommend scheduling an immediate comprehensive mental health counseling evaluation.'}
          </p>
        </div>

        {/* Footer Disclaimer */}
        <div className="flex items-center justify-between text-[11px] text-adaptive-muted pt-2 border-t border-slate-200/20 dark:border-white/10">
          <div className="flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400" />
            <span>Generated: {record.created_at ? new Date(record.created_at).toLocaleString() : new Date().toLocaleString()}</span>
          </div>

          <div className="flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400" />
            <span>Decision Tree 99.5% Acc</span>
          </div>
        </div>

      </div>
    </div>
  );
}
