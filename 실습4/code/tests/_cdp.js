/**
 * _cdp.js — 헤드리스 Chrome/Edge 를 띄우고 CDP 로 붙는 최소 헬퍼.
 *
 * 앞의 하니스들(pose/effects/aim/move)은 THREE 스텁 위에서 순수 로직만 본다.
 * 그것만으로는 "페이지가 실제로 그려지는가"를 못 잡는다 — 실제로 `fx` 이름 충돌로
 * animate()가 매 프레임 예외를 던져 1인칭 화면이 통째로 검은 화면이 된 적이 있는데,
 * 스텁 하니스는 전부 통과했었다. 그래서 진짜 브라우저로 띄우는 경로를 따로 둔다.
 *
 * 외부 의존성 없음 — Node 22의 내장 fetch/WebSocket 만 쓴다.
 */
const { spawn } = require('child_process');
const path = require('path');

// 브라우저 실행 파일 후보. **플랫폼마다 다르다** — Windows 경로만 두면 리눅스
// 컨테이너(웹 Claude Code · CI)에서 브라우저 하니스가 통째로 못 돈다.
// 우선순위: 환경변수 → 플랫폼별 표준 경로 → PATH 탐색.
const BROWSER_PATHS = {
  win32: [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ],
  darwin: [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  ],
  linux: [
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/microsoft-edge',
    '/snap/bin/chromium',
  ],
};

// PATH 에서 찾을 실행 파일 이름 (경로가 배포판마다 달라도 이쪽으로 걸린다)
const BROWSER_NAMES = [
  'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser',
  'microsoft-edge', 'chrome',
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

function findBrowser() {
  const fs = require('fs');

  // 1) 환경변수로 직접 지정 — 어느 플랫폼이든 이게 최우선
  const env = process.env.CHROME_PATH || process.env.CHROME_BIN || process.env.BROWSER_PATH;
  if (env && fs.existsSync(env)) return env;

  // 2) 플랫폼별 표준 설치 경로
  const candidates = BROWSER_PATHS[process.platform] || BROWSER_PATHS.linux;
  const hit = candidates.find(p => fs.existsSync(p));
  if (hit) return hit;

  // 3) PATH 탐색. 리눅스 컨테이너는 설치 경로가 배포판마다 달라 이쪽이 더 잘 걸린다.
  const dirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean);
  for (const name of BROWSER_NAMES) {
    for (const dir of dirs) {
      const full = path.join(dir, name);
      if (fs.existsSync(full)) return full;
    }
  }

  throw new Error([
    'Chrome/Edge 를 찾지 못했습니다.',
    '  설치돼 있다면 CHROME_PATH 환경변수로 실행 파일을 지정하세요.',
    '  브라우저가 없는 환경(웹 Claude Code 등)이라면 로직 하니스 7종만 돌리면 됩니다 —',
    '  pose / effects / aim / move / face / punch / doc',
  ].join(String.fromCharCode(10)));
}

/**
 * 헤드리스 브라우저를 띄우고 CDP 세션을 연다.
 * @returns {{ evaluate, logs, close }}
 *   evaluate(expr) — 페이지 안에서 식을 평가해 값을 반환 (예외는 { ERROR } 로)
 *   logs           — 수집된 콘솔/예외 배열 (참조를 그대로 들고 있으면 계속 쌓인다)
 */
async function open(url, { port = 9333, fakeMedia = true, settle = 6000 } = {}) {
  const args = [
    '--headless=new', `--remote-debugging-port=${port}`,
    '--ignore-certificate-errors', '--allow-insecure-localhost',
    // WebGL 을 소프트웨어로 — CI/원격 데스크톱에도 GPU가 없을 수 있다
    '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
    '--window-size=1280,800', '--no-sandbox', '--disable-dev-shm-usage', 'about:blank',
  ];
  if (fakeMedia) args.splice(3, 0, '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream');

  const proc = spawn(findBrowser(), args, { stdio: 'ignore' });

  let targets = null;
  for (let i = 0; i < 40; i++) {
    try {
      targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      if (targets.length) break;
    } catch (e) { /* 아직 안 떴다 */ }
    await sleep(300);
  }
  if (!targets || !targets.length) { proc.kill(); throw new Error('브라우저 CDP 연결 실패'); }

  const page = targets.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const logs = [];

  const send = (method, params) => new Promise(res => {
    const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params: params || {} }));
  });

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      const desc = (d.exception && (d.exception.description || d.exception.value)) || d.text;
      logs.push({ kind: 'EXCEPTION', text: String(desc).split('\n').slice(0, 3).join(' | ') });
    } else if (m.method === 'Runtime.consoleAPICalled') {
      const t = m.params.args.map(a => a.value !== undefined ? a.value : (a.description || a.type)).join(' ');
      logs.push({ kind: m.params.type.toUpperCase(), text: String(t).slice(0, 300) });
    } else if (m.method === 'Log.entryAdded') {
      logs.push({ kind: 'LOG:' + m.params.entry.level, text: String(m.params.entry.text).slice(0, 300) });
    }
  };
  await new Promise(r => ws.onopen = r);
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url });
  await sleep(settle);

  return {
    logs,
    async evaluate(expression) {
      const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) {
        return { ERROR: (r.exceptionDetails.exception && r.exceptionDetails.exception.description) || r.exceptionDetails.text };
      }
      return r.result.value;
    },
    /** 페이지를 PNG 로 저장한다 (문서·발표용 캡처). */
    async screenshot(filePath, opts) {
      const r = await send('Page.captureScreenshot', Object.assign({ format: 'png' }, opts || {}));
      if (!r || !r.data) return false;
      require('fs').writeFileSync(filePath, Buffer.from(r.data, 'base64'));
      return true;
    },
    close() { try { ws.close(); } catch (e) {} proc.kill(); },
  };
}

/** 무시해도 되는 잡음 (자체서명 인증서 경고, favicon 404, WebGL 드라이버 메시지 등) */
function isNoise(l) {
  return /valid SSL certificate|favicon|GL Driver Message|gl_context|Successfully created a WebGL|OpenGL error checking|swiftshader|slot_in_use/i.test(l.text);
}

module.exports = { open, sleep, isNoise, findBrowser };
