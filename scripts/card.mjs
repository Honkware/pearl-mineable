// Renders the built site with live chain data into the link-preview image, then points
// index.html at a file named for this build: Discord and X cache an image by its URL.
//   node scripts/card.mjs <site url> <dist dir> [chrome path]
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, dist, chromePath = 'google-chrome'] = process.argv.slice(2);
const chrome = spawn(chromePath, ['--headless=new', '--no-sandbox', '--remote-debugging-port=9333', `--user-data-dir=${mkdtempSync(join(tmpdir(), 'card-'))}`, '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await sleep(200);
    target = await fetch('http://127.0.0.1:9333/json').then((r) => r.json()).then((t) => t.find((x) => x.type === 'page')).catch(() => undefined);
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r));
  let id = 0;
  const waits = new Map();
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    waits.get(m.id)?.(m);
    waits.delete(m.id);
  });
  const send = (method, params = {}) => new Promise((r) => { const n = ++id; waits.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
  await send('Emulation.setDeviceMetricsOverride', { width: 1200, height: 630, deviceScaleFactor: 2, mobile: false });
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });
  await send('Page.navigate', { url });
  let loaded = false;
  for (let i = 0; i < 60 && !loaded; i++) {
    await sleep(500);
    const r = await send('Runtime.evaluate', { expression: "document.querySelector('#strip b')?.textContent ?? ''", returnByValue: true });
    loaded = /\d/.test(r.result?.result?.value ?? '');
  }
  // Without chain data the card would show dashes, so the committed image stays.
  if (!loaded) throw new Error('chain data did not load; keeping the committed card');
  // CI runners lack a font with the theme button's glyph.
  await send('Runtime.evaluate', { expression: "document.getElementById('theme')?.remove()" });
  await sleep(1500);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const name = `card-${Date.now().toString(36)}.png`;
  writeFileSync(join(dist, name), Buffer.from(shot.result.data, 'base64'));
  const html = join(dist, 'index.html');
  writeFileSync(html, readFileSync(html, 'utf8').replace(/og\.png"/, `${name}"`));
  console.log(`card ${name}`);
  ws.close();
} finally {
  chrome.kill();
}
