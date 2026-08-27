/**
 * aim_harness.js — 자동 조준 로직 헤드리스 검증
 *
 * fighter_client.html 에서 findNearestOpponent / snapAimForPunch 의 **원문을 그대로 추출**해
 * 구동한다. 테스트에 로직을 베껴 쓰면 실제 파일과 어긋나도 통과해버리므로, 반드시 원본을 읽는다.
 *
 *   cd iter3/tests && node aim_harness.js
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '../server/templates/fighter_client.html'), 'utf8');

function extract(name) {
  const i = html.indexOf(`function ${name}(`);
  if (i < 0) throw new Error(`${name} 를 fighter_client.html 에서 찾지 못했습니다`);
  // 중괄호 균형을 세어 함수 본문 끝을 찾는다
  let depth = 0, started = false, j = i;
  for (; j < html.length; j++) {
    if (html[j] === '{') { depth++; started = true; }
    else if (html[j] === '}') { depth--; if (started && depth === 0) { j++; break; } }
  }
  return html.slice(i, j);
}

// fighter_client.html 과 동일한 스폰 표 (worldConfigs)
const worldConfigs = {
  client_1: { pos: [-12, 0, 0],  yaw:  Math.PI / 2, color: 0xff3366 },
  client_2: { pos: [12, 0, 0],   yaw: -Math.PI / 2, color: 0x00e5ff },
  client_3: { pos: [0, 0, -12],  yaw: 0,            color: 0xffd700 },
  client_4: { pos: [0, 0, 12],   yaw: Math.PI,      color: 0x00ff66 },
};
const startConfigs = {
  client_1: { x: -12, z: 0,   yaw: -Math.PI / 2 },
  client_2: { x: 12,  z: 0,   yaw:  Math.PI / 2 },
  client_3: { x: 0,   z: -12, yaw:  Math.PI },
  client_4: { x: 0,   z: 12,  yaw:  0 },
};

// 추출한 함수가 참조하는 스코프 변수들
let playerX, playerZ, rotationAngle, rotAnchor;
let opponentPositions, opponentMeshes, oppHp;
const AIM_SNAP_RANGE = 11.0;
const AIM_SNAP_MAX_TURN = 120 * Math.PI / 180;
const angleWrap = a => Math.atan2(Math.sin(a), Math.cos(a));

const findNearestOpponent = eval(`(${extract('findNearestOpponent')})`);
const snapAimForPunch     = eval(`(${extract('snapAimForPunch')})`);

/** fighter_client.html 의 초기화와 동일하게 세팅한다 (패킷은 한 개도 받지 않은 상태). */
function setupSolo(myId) {
  opponentMeshes = {};
  Object.keys(worldConfigs).forEach(id => { if (id !== myId) opponentMeshes[id] = {}; });
  // ★ 이번 수정의 핵심: 상대 위치를 스폰 좌표로 미리 채운다
  opponentPositions = {};
  Object.keys(opponentMeshes).forEach(id => {
    const cfg = worldConfigs[id];
    opponentPositions[id] = { x: cfg.pos[0], z: cfg.pos[2], yaw: cfg.yaw };
  });
  oppHp = {};
  Object.keys(opponentMeshes).forEach(id => { oppHp[id] = 100; });
  const st = startConfigs[myId];
  playerX = st.x; playerZ = st.z; rotationAngle = st.yaw; rotAnchor = null;
}

let fail = 0;
const ck = (name, cond, extra) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${extra !== undefined ? `  (${extra})` : ''}`);
  if (!cond) fail++;
};
const deg = r => (r * 180 / Math.PI).toFixed(1);

console.log('--- 혼자 접속 (상대 패킷 0개) — 영상에서 실패했던 조건 ---');
for (const myId of Object.keys(startConfigs)) {
  setupSolo(myId);
  const t = findNearestOpponent();
  ck(`${myId}: 타깃을 찾는다`, t !== null, t ? `${t.id} ${t.dist.toFixed(1)}u` : 'null');
}

console.log('\n--- 펀치 스냅이 실제로 방향을 바꾸는가 ---');
{
  setupSolo('client_1');                 // (-12,0)에서 +x(링 중앙)를 봄
  const before = rotationAngle;
  const t = findNearestOpponent();
  // 현재 시선(+x)과 직각인 -z 쪽 5u 지점에 상대를 놓는다 → 90도를 실제로 돌아야 한다
  opponentPositions[t.id] = { x: playerX, z: playerZ - 5 };
  const snapped = snapAimForPunch();
  ck('스냅이 걸린다', snapped !== null, snapped ? snapped.id : 'null');
  ck('yaw가 실제로 90도 돈다', Math.abs(angleWrap(rotationAngle - before) - Math.PI / 2) < 0.02,
     `${deg(before)}deg -> ${deg(rotationAngle)}deg`);
  const err = angleWrap(Math.atan2(-0, 5) - rotationAngle);
  ck('상대를 정확히 겨눈다 (오차 < 1deg)', Math.abs(err) < 0.017, `${deg(err)}deg`);
  ck('회전 앵커가 풀린다 (45deg 제한이 되감지 않도록)', rotAnchor === null);
}

console.log('\n--- K.O.된 상대는 조준하지 않는가 ---');
{
  setupSolo('client_1');
  const first = findNearestOpponent();
  oppHp[first.id] = 0;
  const second = findNearestOpponent();
  ck('죽은 상대는 건너뛴다', second && second.id !== first.id,
     `${first.id}(사망) -> ${second ? second.id : 'null'}`);
  Object.keys(oppHp).forEach(id => { oppHp[id] = 0; });
  ck('전원 K.O.면 타깃 없음', findNearestOpponent() === null);
}

console.log('\n--- 사거리 / 각도 경계 ---');
{
  const cases = [
    ['정면 6u',      0,   6,  true],
    ['옆 90deg 6u',  90,  6,  true],
    ['119deg 6u',    119, 6,  true],
    ['150deg 6u',    150, 6,  false],
    ['정면 11u',     0,   11, true],
    ['정면 12u',     0,   12, false],
  ];
  for (const [label, angDeg, dist, expect] of cases) {
    setupSolo('client_1');
    rotationAngle = 0;                       // -z 를 봄
    playerX = 0; playerZ = 0;
    const a = angDeg * Math.PI / 180;
    opponentPositions = { client_2: { x: dist * Math.sin(a), z: -dist * Math.cos(a) } };
    oppHp = { client_2: 100 };
    const got = snapAimForPunch() !== null;
    ck(`${label} -> ${expect ? '스냅' : '스냅 안 함'}`, got === expect);
  }
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
