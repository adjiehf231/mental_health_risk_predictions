const os = require('os');
const { spawn } = require('child_process');

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return '127.0.0.1';
}

const localIp = getLocalIp();

console.log('\n================================================================');
console.log('🚀 MENTAL HEALTH RISK PREDICTIONS - LOCAL NETWORK (LAN) RUNNER');
console.log('================================================================\n');
console.log(` 💻 Akses di Laptop Ini    : http://localhost:3000`);
console.log(` 📱 Akses dari HP / Laptop Lain: http://${localIp}:3000`);
console.log('\n⚠️  PENTING: Jangan mengetik http://0.0.0.0:3000 di browser!');
console.log(`   Gunakan alamat IP di atas: http://${localIp}:3000\n`);
console.log('----------------------------------------------------------------\n');

// Spawn next dev -H 0.0.0.0 -p 3000
const nextProc = spawn('npx', ['next', 'dev', '-H', '0.0.0.0', '-p', '3000'], {
  stdio: 'inherit',
  shell: true,
});

nextProc.on('exit', (code) => {
  process.exit(code || 0);
});
