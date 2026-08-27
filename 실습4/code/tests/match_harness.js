/**
 * match_harness.js — 경기 진행/승패 규칙을 실제 arena 페이지에서 검증한다.
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd iter3/tests && node match_harness.js [베이스URL]
 *
 * WebSocket 슬롯(host_arena)이 이미 사용 중이어도 상관없다.
 * `socket.onmessage` 를 페이지 안에서 직접 호출하면 서버 패킷을 받은 것과 똑같은 경로를 탄다.
 * 덕분에 사람이 붙어 있는 상태에서도 K.O. 시나리오를 끝까지 돌려볼 수 있다.
 */
const { open, sleep } = require('./_cdp');
const BASE = process.argv[2] || 'https://localhost:8000';

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

const fightersJson = (hps) => JSON.stringify({
  client_1: { name: 'Red Boxer',     hp: hps[0], world_x: -12, world_z: 0,   yaw: -1.57 },
  client_2: { name: 'Cyan Boxer',    hp: hps[1], world_x: 12,  world_z: 0,   yaw: 1.57 },
  client_3: { name: 'Gold Mage',     hp: hps[2], world_x: 0,   world_z: -12, yaw: 3.14 },
  client_4: { name: 'Green Striker', hp: hps[3], world_x: 0,   world_z: 12,  yaw: 0 },
});

(async () => {
  const p = await open(`${BASE}/arena`, { port: 9343, settle: 6000, fakeMedia: false });
  const feed = (hps) => p.evaluate(`socket.onmessage({data: JSON.stringify({fighters: ${fightersJson(hps)}})})`);
  const modal = () => p.evaluate(`document.getElementById('result-modal').style.display`);
  const clock = () => p.evaluate(`document.getElementById('round-clock').innerText`);
  const toSec = (s) => { const [m, x] = String(s).split(':').map(Number); return m * 60 + x; };

  console.log('--- 제한 시간 없음: 시계는 경과 시간만 센다 ---');
  const t1 = await clock();
  await sleep(3200);
  const t2 = await clock();
  ck('mm:ss 형식', /^\d+:\d\d$/.test(t2), `${t1} -> ${t2}`);
  ck('시간이 증가한다 (카운트다운 아님)', toSec(t2) > toSec(t1), `${toSec(t1)}s -> ${toSec(t2)}s`);
  ck('시간으로는 경기가 끝나지 않는다', (await modal()) !== 'flex');
  ck('"TIME OVER" 판정승 문구가 없다', !(await p.evaluate(`document.body.innerText.includes('TIME OVER')`)));

  console.log('');
  console.log('--- 승패: 마지막 1인이 남을 때만 종료 ---');
  await feed([100, 100, 100, 100]);
  ck('4명 생존 → 계속', (await modal()) !== 'flex');
  await feed([40, 0, 100, 100]);
  ck('1명 K.O. → 계속', (await modal()) !== 'flex');
  await feed([40, 0, 0, 100]);
  ck('2명 K.O. → 계속', (await modal()) !== 'flex');
  await feed([37, 0, 0, 0]);
  await sleep(400);
  ck('3명 K.O. → 승자 선언', (await modal()) === 'flex');

  const title = await p.evaluate(`document.getElementById('result-winner-title').innerText`);
  const sub = await p.evaluate(`document.getElementById('result-sub').innerText`);
  ck('생존자가 승자로 표시된다', String(title).includes('Red Boxer'), title);
  ck('사유가 LAST MAN STANDING', String(sub).includes('LAST MAN STANDING'), sub);

  console.log('');
  console.log('--- K.O. 표시 ---');
  ck('K.O.된 카드는 흑백 처리', await p.evaluate(`document.getElementById('card-client_2').classList.contains('ko')`));
  ck('생존자 카드는 정상', !(await p.evaluate(`document.getElementById('card-client_1').classList.contains('ko')`)));
  ck('K.O.된 아바타는 링에서 사라진다',
     await p.evaluate(`(() => {
       const m = fighterMeshes['client_2'];
       return !!m && (m.isDown() === true);
     })()`));
  ck('생존자 아바타는 링에 남아 있다',
     await p.evaluate(`fighterMeshes['client_1'].isDown() === false`));

  const errs = p.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', errs.length === 0, errs.map(e => e.text).join(' / ') || 'none');

  p.close();
  console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('하니스 오류:', e && e.message); process.exit(1); });
