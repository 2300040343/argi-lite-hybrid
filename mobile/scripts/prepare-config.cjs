const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const input = path.join(root, 'download-config.json');
const config = fs.existsSync(input)
  ? JSON.parse(fs.readFileSync(input, 'utf8'))
  : { urls: {}, expiresAt: null };
fs.writeFileSync(
  path.join(root, 'download-config.generated.json'),
  JSON.stringify(config, null, 2) + '\n',
  { mode: 0o600 },
);
fs.chmodSync(path.join(root, 'download-config.generated.json'), 0o600);
