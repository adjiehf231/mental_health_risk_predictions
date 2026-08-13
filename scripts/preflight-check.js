/**
 * Pre-flight Production Release Verification Script
 * Validates build readiness, environment setup, and test suite compliance before Vercel & Supabase deployment.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Running Pre-flight Production Release Verification...\n');

let failed = false;

function runStep(name, command) {
  process.stdout.write(`⏳ Checking ${name}... `);
  try {
    execSync(command, { stdio: 'pipe' });
    console.log('✅ PASSED');
  } catch (error) {
    console.log('❌ FAILED');
    console.error(`   Error details: ${error.message.split('\n')[0]}`);
    failed = true;
  }
}

// 1. Verify Core Manifests & Config Files
const requiredFiles = [
  'package.json',
  'next.config.js',
  'tailwind.config.js',
  'tsconfig.json',
  'supabase_schema.sql',
  'vercel.json',
  'PRD.md',
  'guide_deploy.md',
  'qa_automation.md',
  'api/py/index.py',
  'api/py/requirements.txt',
  'models/best_model.pkl',
];

process.stdout.write('⏳ Checking Core Project Manifests... ');
const missingFiles = requiredFiles.filter(f => !fs.existsSync(path.join(__dirname, '..', f)));

if (missingFiles.length === 0) {
  console.log('✅ PASSED (All files present)');
} else {
  console.log(`❌ FAILED (Missing files: ${missingFiles.join(', ')})`);
  failed = true;
}

// 2. Run Vitest Unit Tests
runStep('Vitest Unit Test Suite', 'npm run test');

// 3. Validate Next.js Production Build
runStep('Next.js Production Build', 'npm run build');

console.log('\n---------------------------------------------------');
if (failed) {
  console.error('❌ Pre-flight Release Verification FAILED. Please resolve the issues above before deploying.');
  process.exit(1);
} else {
  console.log('🎉 Pre-flight Verification SUCCESSFUL! Platform is 100% ready for Vercel & Supabase production release.');
  process.exit(0);
}
