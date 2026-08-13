'use client';

import { Cpu, CheckCircle2, Zap, ShieldCheck, Award, LineChart as LineChartIcon, Lightbulb } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import { useApp } from '@/lib/AppContext';

export default function ModelsPage() {
  const { t } = useApp();

  // 1. Model Evaluation Metrics Data
  const modelMetrics = [
    { name: 'C4.5 Decision Tree', accuracy: 99.5, f1: 99.3, precision: 99.4, recall: 99.2, auc: 0.998, isBest: true, color: '#10b981' },
    { name: 'Random Forest', accuracy: 97.2, f1: 96.8, precision: 97.0, recall: 96.6, auc: 0.985, isBest: false, color: '#6366f1' },
    { name: 'SVM (RBF Kernel)', accuracy: 93.5, f1: 92.9, precision: 93.1, recall: 92.7, auc: 0.942, isBest: false, color: '#8b5cf6' },
    { name: 'KNN (k=10)', accuracy: 91.8, f1: 91.2, precision: 91.5, recall: 90.9, auc: 0.926, isBest: false, color: '#ec4899' },
    { name: 'Naive Bayes (Gaussian)', accuracy: 89.4, f1: 88.7, precision: 89.0, recall: 88.4, auc: 0.898, isBest: false, color: '#06b6d4' },
  ];

  // 2. Feature Importance Data
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

  // 3. ROC-AUC Curve Synthetic Points (FPR vs TPR coordinates for all 5 models + baseline)
  const rocCurveData = [
    { fpr: 0.00, decisionTree: 0.00, randomForest: 0.00, svm: 0.00, knn: 0.00, naiveBayes: 0.00, randomGuess: 0.00 },
    { fpr: 0.02, decisionTree: 0.94, randomForest: 0.88, svm: 0.72, knn: 0.65, naiveBayes: 0.58, randomGuess: 0.02 },
    { fpr: 0.05, decisionTree: 0.98, randomForest: 0.94, svm: 0.83, knn: 0.78, naiveBayes: 0.70, randomGuess: 0.05 },
    { fpr: 0.10, decisionTree: 0.99, randomForest: 0.97, svm: 0.90, knn: 0.85, naiveBayes: 0.79, randomGuess: 0.10 },
    { fpr: 0.20, decisionTree: 1.00, randomForest: 0.99, svm: 0.95, knn: 0.91, naiveBayes: 0.86, randomGuess: 0.20 },
    { fpr: 0.40, decisionTree: 1.00, randomForest: 1.00, svm: 0.98, knn: 0.96, naiveBayes: 0.92, randomGuess: 0.40 },
    { fpr: 0.60, decisionTree: 1.00, randomForest: 1.00, svm: 0.99, knn: 0.98, naiveBayes: 0.96, randomGuess: 0.60 },
    { fpr: 0.80, decisionTree: 1.00, randomForest: 1.00, svm: 1.00, knn: 0.99, naiveBayes: 0.98, randomGuess: 0.80 },
    { fpr: 1.00, decisionTree: 1.00, randomForest: 1.00, svm: 1.00, knn: 1.00, naiveBayes: 1.00, randomGuess: 1.00 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-600 dark:text-emerald-300 border border-emerald-500/20">
          <Cpu className="w-4 h-4 text-emerald-500 dark:text-emerald-400" />
          <span>{t.models.badge}</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-adaptive-white">
          🤖 {t.models.title}
        </h1>
        <p className="text-adaptive-muted max-w-2xl mx-auto text-sm sm:text-base">
          {t.models.subtitle}
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
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">{t.models.topTitle}</span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-500 text-white uppercase">
                {t.models.topWinner}
              </span>
            </div>
            <h2 className="text-2xl font-extrabold text-adaptive-white mt-0.5">{t.models.topModel}</h2>
            <p className="text-xs text-adaptive-muted mt-1">
              {t.models.topSub}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 sm:gap-6 text-center">
          <div className="p-3 rounded-2xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/10 px-5">
            <span className="text-xs text-adaptive-muted font-medium block">Accuracy</span>
            <span className="text-3xl font-extrabold text-emerald-600 dark:text-emerald-400">99.5%</span>
          </div>
          <div className="p-3 rounded-2xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/10 px-5">
            <span className="text-xs text-adaptive-muted font-medium block">ROC-AUC</span>
            <span className="text-3xl font-extrabold text-emerald-600 dark:text-emerald-400">0.998</span>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="glass-panel rounded-3xl border border-slate-200/40 dark:border-white/10 overflow-hidden">
        <div className="p-6 border-b border-slate-200/20 dark:border-white/10">
          <h3 className="text-lg font-bold text-adaptive-white flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            {t.models.tableTitle}
          </h3>
          <p className="text-xs text-adaptive-muted">{t.models.tableSub}</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-adaptive-white">
            <thead className="bg-slate-200/40 dark:bg-white/5 text-xs text-adaptive-muted uppercase tracking-wider border-b border-slate-200/40 dark:border-white/10">
              <tr>
                <th className="px-6 py-4">{t.models.columns.algo}</th>
                <th className="px-6 py-4">{t.models.columns.acc}</th>
                <th className="px-6 py-4">{t.models.columns.f1}</th>
                <th className="px-6 py-4">{t.models.columns.prec}</th>
                <th className="px-6 py-4">{t.models.columns.rec}</th>
                <th className="px-6 py-4">ROC-AUC</th>
                <th className="px-6 py-4 text-right">{t.models.columns.status}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200/30 dark:divide-white/5">
              {modelMetrics.map((m, idx) => (
                <tr key={idx} className={m.isBest ? 'bg-emerald-500/10 font-medium' : 'hover:bg-black/5 dark:hover:bg-white/5 transition-colors'}>
                  <td className="px-6 py-4 flex items-center gap-2 font-bold text-adaptive-white">
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: m.color }} />
                    {m.name}
                  </td>
                  <td className="px-6 py-4 font-semibold text-adaptive-white">{m.accuracy}%</td>
                  <td className="px-6 py-4">{m.f1}%</td>
                  <td className="px-6 py-4">{m.precision}%</td>
                  <td className="px-6 py-4">{m.recall}%</td>
                  <td className="px-6 py-4 font-extrabold text-indigo-600 dark:text-indigo-400">{m.auc}</td>
                  <td className="px-6 py-4 text-right">
                    {m.isBest ? (
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/20 px-2.5 py-1 rounded-full border border-emerald-500/30">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Best Model
                      </span>
                    ) : (
                      <span className="text-xs text-adaptive-muted">Evaluated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ROC-AUC Curve Section */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-6">
        
        <div className="border-b border-slate-200/20 dark:border-white/10 pb-4">
          <h3 className="text-xl font-extrabold text-adaptive-white flex items-center gap-2">
            <LineChartIcon className="w-6 h-6 text-indigo-500 dark:text-indigo-400" />
            {t.models.rocTitle}
          </h3>
          <p className="text-xs text-adaptive-muted mt-1">{t.models.rocSub}</p>
        </div>

        {/* ROC AUC Summary Score Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {modelMetrics.map((m, i) => (
            <div key={i} className="p-3.5 rounded-2xl bg-black/5 dark:bg-white/5 border border-slate-200/40 dark:border-white/5 text-center space-y-1">
              <span className="text-[10px] font-bold text-adaptive-muted uppercase tracking-wider block truncate">{m.name}</span>
              <p className="text-xl font-black" style={{ color: m.color }}>AUC {m.auc}</p>
            </div>
          ))}
        </div>

        {/* Recharts ROC Line Chart */}
        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rocCurveData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                tickCount={6}
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: 'False Positive Rate (FPR)', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 12 }}
              />
              <YAxis
                domain={[0, 1]}
                tickCount={6}
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', offset: 10, fill: '#94a3b8', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }}
                formatter={(val: any, name: any) => [`TPR: ${val}`, name]}
                labelFormatter={(fpr: any) => `FPR: ${fpr}`}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
              <Line type="monotone" dataKey="decisionTree" name="Decision Tree (AUC: 0.998)" stroke="#10b981" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="randomForest" name="Random Forest (AUC: 0.985)" stroke="#6366f1" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="svm" name="SVM RBF (AUC: 0.942)" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="knn" name="KNN k=10 (AUC: 0.926)" stroke="#ec4899" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="naiveBayes" name="Naive Bayes (AUC: 0.898)" stroke="#06b6d4" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="randomGuess" name="Random Guess (AUC: 0.500)" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Technical Interpretation Box */}
        <div className="p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-adaptive-white space-y-1.5">
          <span className="font-bold text-indigo-600 dark:text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
            <Lightbulb className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
            {t.models.rocInsightTitle}
          </span>
          <p className="leading-relaxed text-adaptive-muted">
            {t.models.rocInsightDesc}
          </p>
        </div>

      </div>

      {/* Feature Importance Section */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-200/40 dark:border-white/10 space-y-4">
        <div className="border-b border-slate-200/20 dark:border-white/10 pb-3">
          <h3 className="text-lg font-bold text-adaptive-white flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-500 dark:text-amber-400" />
            {t.models.featureTitle}
          </h3>
          <p className="text-xs text-adaptive-muted">{t.models.featureSub}</p>
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
