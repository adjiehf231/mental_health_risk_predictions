'use client';

import { Cpu, CheckCircle2, Zap, Layers, ShieldCheck, Award } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ModelsPage() {
  const modelMetrics = [
    { name: 'C4.5 Decision Tree', accuracy: 99.5, f1: 99.3, precision: 99.4, recall: 99.2, isBest: true, color: '#10b981' },
    { name: 'Random Forest', accuracy: 97.2, f1: 96.8, precision: 97.0, recall: 96.6, isBest: false, color: '#6366f1' },
    { name: 'SVM (RBF Kernel)', accuracy: 93.5, f1: 92.9, precision: 93.1, recall: 92.7, isBest: false, color: '#8b5cf6' },
    { name: 'KNN (k=10)', accuracy: 91.8, f1: 91.2, precision: 91.5, recall: 90.9, isBest: false, color: '#ec4899' },
    { name: 'Naive Bayes (Gaussian)', accuracy: 89.4, f1: 88.7, precision: 89.0, recall: 88.4, isBest: false, color: '#06b6d4' },
  ];

  const featureImportanceScores = [
    { feature: 'Depression Score', score: 98.4 },
    { feature: 'Anxiety Score', score: 95.2 },
    { feature: 'Work Stress Level', score: 88.7 },
    { feature: 'Sleep Duration', score: 84.1 },
    { feature: 'Financial Stress', score: 79.5 },
    { feature: 'Social Support', score: 74.2 },
    { feature: 'Job Satisfaction', score: 68.9 },
    { feature: 'Panic Attack History', score: 64.3 },
    { feature: 'Screen Time', score: 58.1 },
    { feature: 'Physical Activity', score: 52.6 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
          <Cpu className="w-4 h-4 text-emerald-400" />
          <span>Cross-Validation Machine Learning Benchmarks</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white">
          🤖 Algorithm Performance & Model Explorer
        </h1>
        <p className="text-slate-300 max-w-2xl mx-auto text-sm sm:text-base">
          Comparison of 5 machine learning models evaluated using 5-fold cross-validation on Kaggle dataset.
        </p>
      </div>

      {/* Best Model Showcase Card */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-emerald-500/30 bg-emerald-500/5 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-600 flex items-center justify-center text-white shadow-lg shadow-emerald-500/30">
            <Award className="w-8 h-8" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">Top Performing Model</span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-500 text-white uppercase">
                WINNER
              </span>
            </div>
            <h2 className="text-2xl font-extrabold text-white mt-0.5">C4.5 Decision Tree Classifier</h2>
            <p className="text-xs text-slate-300 mt-1">
              Max Depth: 10 | Criterion: Gini | K-Fold Cross Validation: 5
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-center">
          <div className="p-3 rounded-2xl bg-white/5 border border-white/10 px-5">
            <span className="text-xs text-slate-400 font-medium block">Accuracy</span>
            <span className="text-3xl font-extrabold text-emerald-400">99.5%</span>
          </div>
          <div className="p-3 rounded-2xl bg-white/5 border border-white/10 px-5">
            <span className="text-xs text-slate-400 font-medium block">F1-Score</span>
            <span className="text-3xl font-extrabold text-emerald-400">99.3%</span>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="glass-panel rounded-3xl border border-white/10 overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-indigo-400" />
            5-Algorithm Evaluation Matrix
          </h3>
          <p className="text-xs text-slate-400">Evaluating 25,000 observations across key classification metrics</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-200">
            <thead className="bg-white/5 text-xs text-slate-400 uppercase tracking-wider border-b border-white/10">
              <tr>
                <th className="px-6 py-4">Algorithm Model</th>
                <th className="px-6 py-4">Accuracy (%)</th>
                <th className="px-6 py-4">F1-Score (%)</th>
                <th className="px-6 py-4">Precision (%)</th>
                <th className="px-6 py-4">Recall (%)</th>
                <th className="px-6 py-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {modelMetrics.map((m, idx) => (
                <tr key={idx} className={m.isBest ? 'bg-emerald-500/10 font-medium' : 'hover:bg-white/5'}>
                  <td className="px-6 py-4 flex items-center gap-2 font-bold text-white">
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: m.color }} />
                    {m.name}
                  </td>
                  <td className="px-6 py-4 text-white font-semibold">{m.accuracy}%</td>
                  <td className="px-6 py-4">{m.f1}%</td>
                  <td className="px-6 py-4">{m.precision}%</td>
                  <td className="px-6 py-4">{m.recall}%</td>
                  <td className="px-6 py-4 text-right">
                    {m.isBest ? (
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-400 bg-emerald-500/20 px-2.5 py-1 rounded-full border border-emerald-500/30">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Best Model
                      </span>
                    ) : (
                      <span className="text-xs text-slate-400">Evaluated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance Section */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-white/10 space-y-4">
        <div className="border-b border-white/10 pb-3">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            Top Feature Importance Scores (SelectKBest f_classif)
          </h3>
          <p className="text-xs text-slate-400">Feature impact scores contributing to mental health risk classification</p>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featureImportanceScores} layout="vertical" margin={{ top: 10, right: 20, left: 40, bottom: 0 }}>
              <XAxis type="number" stroke="#94a3b8" fontSize={11} />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} width={130} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }} />
              <Bar dataKey="score" fill="#6366f1" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
