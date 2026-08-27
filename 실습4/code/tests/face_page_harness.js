/**
 * face_page_harness.js — 3D 얼굴이 실제 페이지에서 "보이는가"를 확인한다.
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd iter3/tests && node face_page_harness.js [베이스URL]
 *
 * 로직 하니스로는 못 잡는 두 가지를 본다.
 *   1) host(arena) 화면이 얼굴 패킷을 받아 실제로 아바타에 적용하는가
 *   2) 적용된 얼굴이 두개골 구에 파묻히지 않는가 — 실제 Face Mesh 토폴로지·비율 기준으로
 *
 * 실제로 겪은 버그: 얼굴이 구 안쪽에 배치돼 host·1인칭 양쪽에서 아무것도 보이지 않았다.
 */
const { open, sleep } = require('./_cdp');
// 머리 반지름은 humanoid.js 에서 읽는다 — 비율을 바꿔도 테스트가 따라오도록
const HEAD_R = Number(
  require('fs').readFileSync(require('path').join(__dirname, '../server/static/humanoid.js'), 'utf8')
    .match(/const HEAD_R = ([\d.]+)/)[1]);
const BASE = process.argv[2] || 'https://localhost:8000';   // 다른 하니스와 동일하게

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

/** 페이지 안에서 사람 얼굴 비슷한 468 랜드마크를 만들어 face blob 을 구성한다. */
const MAKE_BLOB = `(() => {
  const N = 468, AR = 480 / 360;
  const lm = new Array(N);
  for (let i = 0; i < N; i++) {
    const t = (i + 0.5) / N;
    const ring = Math.sqrt(t);                 // 중앙이 촘촘한 분포
    const ang = Math.PI * (1 + Math.sqrt(5)) * i;
    const X = Math.cos(ang) * ring * 0.13;
    const Y = Math.sin(ang) * ring * 0.17;
    const Z = (0.055 * Math.cos(ring * Math.PI * 0.72) - 0.026);
    lm[i] = { x: 0.5 + X / AR, y: 0.5 - Y, z: Z / AR };
  }
  lm[234] = { x: 0.5 - 0.13 / AR, y: 0.5, z: 0 };     // 좌 광대
  lm[454] = { x: 0.5 + 0.13 / AR, y: 0.5, z: 0 };     // 우 광대
  lm[1]   = { x: 0.5, y: 0.505, z: -0.055 / AR };     // 코끝 (앞으로)
  lm[152] = { x: 0.5, y: 0.5 + 0.17, z: 0 };          // 턱
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#c89a6b'; g.fillRect(0, 0, 64, 64);
  return window.serializeFace(lm, c, {
    aspect: AR, imageW: 480, imageH: 360,
    crop: { x0: 100, y0: 40, w: 220, h: 220 },
  }, 0.7);
})()`;

/** 얼굴 정점이 두개골 타원체 밖에 있는지 세는 코드 (페이지 안에서 실행) */
const CHECK_BURIED = (meshExpr) => `(() => {
  const h = ${meshExpr};
  const f = h && h.getFace && h.getFace();
  if (!f) return { applied: false };
  const pos = f.mesh.geometry.attributes.position.array;
  const dz = f.mesh.position.z;
  const fk = f.mesh.scale.x;                       // 머리 크기에 맞춘 스케일
  const R = ${HEAD_R}, sx = h.head.scale.x, sy = h.head.scale.y, sz = h.head.scale.z;
  let inside = 0, worst = Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i] * fk, y = pos[i + 1] * fk, z = pos[i + 2] * fk + dz;
    const e = (x/(R*sx))**2 + (y/(R*sy))**2 + (z/(R*sz))**2;
    if (e < 1) inside++;
    if (e < worst) worst = e;
  }
  return {
    applied: true, inside, worst, faceZ: dz,
    tris: f.triangleCount,
    fullHead: !!f.isFullHead,
    headVisible: h.head.visible,
    faceInScene: !!f.mesh.parent,
    faceVisible: f.mesh.visible !== false,
    skull: [sx, sy, sz],
  };
})()`;

(async () => {
  // ── Host (arena) ─────────────────────────────────────────────
  console.log('--- Host 관제 화면 ---');
  const a = await open(`${BASE}/arena`, { port: 9371, settle: 8000, fakeMedia: false });

  ck('face3d.js 가 로드된다', (await a.evaluate(`typeof window.createFace3D`)) === 'function');
  ck('Face Mesh 토폴로지가 있다', (await a.evaluate(
    `(typeof FACEMESH_TESSELATION !== 'undefined') ? FACEMESH_TESSELATION.length : 0`)) > 0);

  // 서버가 보낸 것처럼 얼굴 패킷을 주입한다
  await a.evaluate(`window.__blob = ${MAKE_BLOB}; socket.onmessage({ data: JSON.stringify({
    type: 'face_update', client_id: 'client_1', face: window.__blob }) }); true`);
  await sleep(1500);

  const hostRes = await a.evaluate(CHECK_BURIED(`fighterMeshes['client_1']`));
  ck('host 가 얼굴 패킷을 받아 아바타에 적용한다', hostRes.applied === true,
     hostRes.applied ? `${hostRes.tris} 삼각형` : '적용 안 됨');
  if (hostRes.applied) {
    ck('얼굴 메쉬가 씬에 붙어 있다', hostRes.faceInScene && hostRes.faceVisible);
    ck('얼굴 삼각형이 canonical 값 이상 (두개골 포함)', hostRes.tris > 852, `${hostRes.tris}`);
    if (hostRes.fullHead) {
      // 닫힌 머리에서는 구를 숨기므로 "구에 파묻힌다"는 개념 자체가 없다.
      // 대신 아래 '닫힌 머리' 절에서 깊이·정점 수를 본다.
      ck('닫힌 머리이므로 구 머리를 숨긴다', hostRes.headVisible === false);
    } else {
      ck('(폴백) 두개골 구에 파묻히지 않는다', hostRes.inside === 0,
         `${hostRes.inside}개 박힘 · 최소비율 ${hostRes.worst.toFixed(2)}`);
      ck('(폴백) 두개골 구는 남아 있다', hostRes.headVisible === true);
    }
  }

  console.log('');
  console.log('--- 닫힌 머리 (앞·옆·뒤 어디서 봐도 사람) ---');
  const head = await a.evaluate(`(() => {
    const h = fighterMeshes['client_1'];
    const f = h.getFace();
    if (!f) return { none: true };
    const pos = f.mesh.geometry.attributes.position.array;
    let zMin = Infinity, zMax = -Infinity, xMin = Infinity, xMax = -Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i + 2] < zMin) zMin = pos[i + 2];
      if (pos[i + 2] > zMax) zMax = pos[i + 2];
      if (pos[i] < xMin) xMin = pos[i];
      if (pos[i] > xMax) xMax = pos[i];
    }
    return {
      isFullHead: f.isFullHead,
      tris: f.triangleCount,
      depth: zMax - zMin, width: xMax - xMin,
      skullHidden: h.head.visible === false,
      jawHidden: h.jaw ? h.jaw.visible === false : null,
      side: f.mesh.material.side,
      verts: pos.length / 3,
    };
  })()`);
  ck('뒤통수까지 닫힌 머리로 만들어진다', head.isFullHead === true);
  ck('얼굴(468)보다 정점이 많다 (두개골이 붙었다)', head.verts > 468, `${head.verts}개`);
  ck('머리 깊이가 폭의 절반 이상 (판자가 아니다)', head.depth > head.width * 0.5,
     `깊이 ${head.depth.toFixed(2)} / 폭 ${head.width.toFixed(2)}`);
  ck('구 머리를 숨긴다 (겹치지 않는다)', head.skullHidden === true);
  ck('턱 덩어리도 숨긴다', head.jawHidden === true);
  ck('앞면만 렌더한다 (안쪽이 비치지 않게)', head.side === 0, `side=${head.side}`);

  // face_bulk (나중에 접속한 host 가 기존 얼굴들을 한 번에 받는 경로)
  await a.evaluate(`socket.onmessage({ data: JSON.stringify({
    type: 'face_bulk', faces: { client_3: window.__blob } }) }); true`);
  await sleep(1200);
  const bulk = await a.evaluate(CHECK_BURIED(`fighterMeshes['client_3']`));
  ck('face_bulk 로 받은 얼굴도 적용된다', bulk.applied === true);

  // HP 를 낮추면 얼굴 표정이 따라가는가
  const expr = await a.evaluate(`(() => {
    const f = fighterMeshes['client_1'].getFace();
    if (!f) return null;
    const before = f.state.bloodAmt;
    socket.onmessage({ data: JSON.stringify({ fighters: {
      client_1: { name:'x', hp: 12, world_x:-12, world_z:0, yaw:0 },
      client_2: { name:'x', hp: 100, world_x:12, world_z:0, yaw:0 },
      client_3: { name:'x', hp: 100, world_x:0, world_z:-12, yaw:0 },
      client_4: { name:'x', hp: 100, world_x:0, world_z:12, yaw:0 } } }) });
    return { before, after: f.state.bloodAmt, hp: f.state.hp };
  })()`);
  ck('HP 가 얼굴에 전달된다 (코피)',
     expr && expr.after > 0.8 && expr.before === 0,
     expr ? `bloodAmt ${expr.before} → ${Number(expr.after).toFixed(2)}` : '얼굴 없음');

  const aerrs = a.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', aerrs.length === 0, aerrs.map(e => e.text).join(' | ') || 'none');
  a.close();
  await sleep(700);

  // ── Fighter (1인칭) ──────────────────────────────────────────
  console.log('');
  console.log('--- Fighter 1인칭 화면 ---');
  const f = await open(`${BASE}/client?id=client_2`, { port: 9372, settle: 9000 });

  await f.evaluate(`window.__blob = ${MAKE_BLOB}; socket.onmessage({ data: JSON.stringify({
    type: 'face_update', client_id: 'client_1', face: window.__blob }) }); true`);
  await sleep(1500);

  const cliRes = await f.evaluate(CHECK_BURIED(`opponentMeshes['client_1']`));
  ck('상대 아바타에 얼굴이 적용된다', cliRes.applied === true,
     cliRes.applied ? `${cliRes.tris} 삼각형` : '적용 안 됨');
  if (cliRes.applied) {
    if (cliRes.fullHead) {
      ck('1인칭에서도 닫힌 머리로 적용된다', cliRes.headVisible === false);
    } else {
      ck('(폴백) 두개골 구에 파묻히지 않는다', cliRes.inside === 0,
         `${cliRes.inside}개 박힘 · 최소비율 ${cliRes.worst.toFixed(2)}`);
      ck('(폴백) 얼굴이 두개골 앞쪽에 놓인다', cliRes.faceZ > 0, cliRes.faceZ.toFixed(2));
    }
  }

  const ferrs = f.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', ferrs.length === 0, ferrs.map(e => e.text).join(' | ') || 'none');
  f.close();

  console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('하니스 오류:', e && e.message); process.exit(1); });
