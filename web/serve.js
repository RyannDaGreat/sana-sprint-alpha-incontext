// Static server for the journal (with the built app at /webgpu/) and the model files at /models/web/.
// HTTPS with a self-signed certificate: WebGPU and Cache Storage need a secure context, and plain
// HTTP on a LAN address is not one. Sends COOP/COEP so ONNX Runtime can use WASM threads.
// Usage: node serve.js [port]
import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import sirv from 'sirv';

const DEFAULT_PORT = 47231;
const CERT_DAYS = 3650;
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CERT_DIR = path.join(ROOT, 'web/.cert');
const port = Number(process.argv[2] ?? DEFAULT_PORT);

/** Command. Creates a self-signed certificate in web/.cert/ on first run; returns {key, cert}. */
function certificate() {
  const key = path.join(CERT_DIR, 'key.pem'), cert = path.join(CERT_DIR, 'cert.pem');
  if (!fs.existsSync(cert)) {
    fs.mkdirSync(CERT_DIR, { recursive: true });
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', cert,
      '-days', String(CERT_DAYS), '-subj', '/CN=sana-web'], { stdio: 'inherit' });
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

const isolation = (res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
};
const opts = { dev: true, etag: true, setHeaders: isolation }; // dev: re-read files, so new model versions show up
const models = sirv(path.join(ROOT, 'models/web'), opts);
const journal = sirv(path.join(ROOT, 'journal'), opts);
const MODELS_PREFIX = '/models/web';

https.createServer(certificate(), (req, res) => {
  console.log(new Date().toISOString(), req.method, req.url, req.headers.range ?? '');
  const notFound = () => { isolation(res); res.statusCode = 404; res.end('not found'); };
  if (req.url.startsWith(MODELS_PREFIX + '/')) { req.url = req.url.slice(MODELS_PREFIX.length); models(req, res, notFound); }
  else journal(req, res, notFound);
}).listen(port, '0.0.0.0', () => console.log(`serving https://0.0.0.0:${port}/webgpu/`));
