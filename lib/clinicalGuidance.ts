import { PatientAssessmentInput } from './types';
import { Language } from './i18n';

export interface MedicalGuidance {
  impression: string;
  specificFindings: string[];
  recommendations: string[];
  urgency: 'routine' | 'moderate' | 'urgent';
}

export function generateMedicalGuidance(
  inputs: PatientAssessmentInput,
  riskPrediction: number,
  lang: Language = 'id'
): MedicalGuidance {
  const isID = lang === 'id';
  const findings: string[] = [];
  const recommendations: string[] = [];

  // 1. Specific Attribute Analysis
  // Sleep Duration
  if (inputs.sleep_hours < 6) {
    findings.push(
      isID
        ? `Deprivasi tidur signifikan (${inputs.sleep_hours} jam/hari). Kurang tidur mendegradasi regulasi emosi korteks prefrontal dan meningkatkan kadar kortisol.`
        : `Significant sleep deprivation detected (${inputs.sleep_hours} hrs/day). Sleep deficit impairs prefrontal emotional regulation and elevates cortisol levels.`
    );
    recommendations.push(
      isID
        ? 'Implementasi protokol higiene tidur medis (jadwal tidur konsisten, kurangi paparan cahaya biru 1 jam sebelum tidur).'
        : 'Implement a structured medical sleep hygiene protocol (consistent schedule, eliminate blue light exposure 1hr prior to sleep).'
    );
  } else if (inputs.sleep_hours >= 7) {
    findings.push(
      isID
        ? `Durasi tidur optimal (${inputs.sleep_hours} jam/hari) mendukung neuroplastisitas dan pemulihan kognitif.`
        : `Optimal sleep duration (${inputs.sleep_hours} hrs/day) supporting neuroplasticity and cognitive recovery.`
    );
  }

  // Anxiety & Depression Scores
  if (inputs.anxiety_score >= 7 || inputs.depression_score >= 7) {
    findings.push(
      isID
        ? `Skor klinis tinggi teridentifikasi (Kecemasan: ${inputs.anxiety_score}/10, Depresi: ${inputs.depression_score}/10), menunjukkan beban afektif berat.`
        : `Elevated clinical scores identified (Anxiety: ${inputs.anxiety_score}/10, Depression: ${inputs.depression_score}/10), indicating substantial affective burden.`
    );
    recommendations.push(
      isID
        ? 'Lakukan skrining psikometri formal berbasis standar medis (skrining GAD-7 untuk kecemasan & PHQ-9 untuk depresi).'
        : 'Conduct formal standardized psychometric screening (GAD-7 for anxiety assessment & PHQ-9 for depression evaluation).'
    );
  } else if (inputs.anxiety_score >= 4 || inputs.depression_score >= 4) {
    findings.push(
      isID
        ? `Skor afektif sedang (Kecemasan: ${inputs.anxiety_score}/10, Depresi: ${inputs.depression_score}/10) mengindikasikan distres emosional subklinis.`
        : `Moderate affective scores (Anxiety: ${inputs.anxiety_score}/10, Depression: ${inputs.depression_score}/10) indicating subclinical emotional distress.`
    );
  }

  // Work & Financial Stress
  if (inputs.work_stress_level >= 7 || inputs.financial_stress_level >= 7) {
    findings.push(
      isID
        ? `Tingkat stres lingkungan tinggi (Stres Kerja: ${inputs.work_stress_level}/10, Stres Keuangan: ${inputs.financial_stress_level}/10) sebagai eksaserbator kelelahan emosional.`
        : `High environmental stress detected (Work Stress: ${inputs.work_stress_level}/10, Financial Stress: ${inputs.financial_stress_level}/10) acting as a major burnout driver.`
    );
    recommendations.push(
      isID
        ? 'Terapkan teknik manajemen stres kognitif dan evaluasi strategi pembagian beban kerja.'
        : 'Apply cognitive stress reduction interventions and workload boundary restructuring.'
    );
  }

  // Panic Attack History & Family History
  if (inputs.panic_attack_history === 1) {
    findings.push(
      isID
        ? 'Riwayat serangan panik positif mengindikasikan reaktivitas sistem saraf otonom yang tinggi.'
        : 'Positive history of panic attacks indicating heightened autonomic nervous system arousal.'
    );
    recommendations.push(
      isID
        ? 'Latih teknik pernapasan diafragma (box breathing) dan grounding untuk regulasi serangan panik.'
        : 'Practice diaphragmatic box breathing and somatic grounding techniques for panic episode regulation.'
    );
  }

  if (inputs.family_history_mental_illness === 1) {
    findings.push(
      isID
        ? 'Predisposisi genetik/keluarga terhadap gangguan kesehatan mental terdeteksi.'
        : 'Genetic/familial predisposition for mental health conditions identified.'
    );
  }

  // Substance Use
  if (inputs.substance_use === 1) {
    findings.push(
      isID
        ? 'Penggunaan zat dilaporkan; berisiko menjadi mekanisme koping maladaptif.'
        : 'Reported substance use; risk of developing maladaptive emotional coping mechanisms.'
    );
    recommendations.push(
      isID
        ? 'Jadwalkan konseling psikoedukasi terkait strategi koping sehat tanpa zat.'
        : 'Schedule psychoeducational counseling focused on healthy non-substance coping strategies.'
    );
  }

  // Fallback findings if healthy
  if (findings.length === 0) {
    findings.push(
      isID
        ? 'Seluruh parameter profil pasien (tidur, stres, afektif) berada dalam batas normal dan seimbang.'
        : 'All patient profile parameters (sleep, stress, affective scores) reside within healthy normal boundaries.'
    );
  }

  if (recommendations.length === 0) {
    recommendations.push(
      isID
        ? 'Pertahankan pola hidup sehat, kurangi screen time malam hari, dan jaga jaringan dukungan sosial.'
        : 'Maintain healthy lifestyle habits, regulate late-night screen time, and foster strong social support networks.'
    );
  }

  // 2. Risk Level Specific Impression & Urgency
  if (riskPrediction === 2) {
    return {
      impression: isID
        ? `PASIEN KATEGORI RISIKO TINGGI (SKOR RISIKO: 2): Teridentifikasi indikator distres psikologis berat dengan kombinasi beban afektif tinggi (${inputs.anxiety_score}/10 kecemasan, ${inputs.depression_score}/10 depresi) dan deprivasi tidur.`
        : `HIGH RISK CATEGORY (RISK SCORE: 2): Significant severe psychological distress identified driven by high affective load (${inputs.anxiety_score}/10 anxiety, ${inputs.depression_score}/10 depression) and sleep deficits.`,
      specificFindings: findings,
      recommendations: [
        isID
          ? 'RUJUKAN KLINIS SEGERA: Disarankan penjadwalan evaluasi tatap muka dengan Psikiater atau Psikolog Klinis.'
          : 'IMMEDIATE CLINICAL REFERRAL: Recommend scheduling an in-person consultation with a Psychiatrist or Clinical Psychologist.',
        ...recommendations,
      ],
      urgency: 'urgent',
    };
  } else if (riskPrediction === 1) {
    return {
      impression: isID
        ? `PASIEN KATEGORI RISIKO SEDANG (SKOR RISIKO: 1): Terdeteksi distres afektif sedang dan tingkat stres kerja (${inputs.work_stress_level}/10) yang memerlukan intervensi dini sebelum berkembang menjadi depresi berat.`
        : `MODERATE RISK CATEGORY (RISK SCORE: 1): Subclinical affective distress and occupational stress (${inputs.work_stress_level}/10) detected, requiring early intervention to prevent severe burnout.`,
      specificFindings: findings,
      recommendations: [
        isID
          ? 'KONSULESI KONSLEING: Disarankan sesi konseling psikologis 2–4 minggu untuk regulasi emosi.'
          : 'COUNSELING CONSULTATION: Recommend 2–4 weeks of psychological counseling sessions for emotional regulation.',
        ...recommendations,
      ],
      urgency: 'moderate',
    };
  } else {
    return {
      impression: isID
        ? `PASIEN KATEGORI RISIKO RENDAH (SKOR RISIKO: 0): Profil psikologis pasien tergolong stabil dan sehat. Hambatan tidur dan stres berada dalam tingkat toleransi adaptif.`
        : `LOW RISK CATEGORY (RISK SCORE: 0): Patient psychological profile is stable and healthy. Sleep and stress parameters reside within adaptive tolerance thresholds.`,
      specificFindings: findings,
      recommendations: [
        isID
          ? 'PEMELIHARAAN PREVENTIF: Lanjutkan gaya hidup seimbang, olahraga teratur, dan higiene tidur.'
          : 'PREVENTIVE MAINTENANCE: Continue balanced lifestyle routines, regular exercise, and healthy sleep hygiene.',
        ...recommendations,
      ],
      urgency: 'routine',
    };
  }
}
