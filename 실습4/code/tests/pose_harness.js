/**
 * pose_harness.js — humanoid.js 절차적 애니메이션 헤드리스 검증
 *
 * 브라우저·Three.js 없이 THREE 최소 스텁(three_stub.js) 위에서 humanoid.js를 직접 구동해
 * 기술별 펀치 궤적 / K.O. 다운 / 피격 리액션이 의도대로 나오는지 수치로 확인한다.
 * 포즈 로직을 고칠 때마다 돌릴 것.
 *
 *   cd iter3/tests && node pose_harness.js
 */
require('./three_stub.js');
require('../server/static/humanoid.js');

let T = 0;
global.performance.now = () => T;

function fresh() { T = 0; return window.createHumanoid(0xff3366); }
function step(h, ms) { T += ms; h.update(); }

// 펀치 궤적을 샘플링해 (어깨X, 어깨Z, 팔꿈치X)의 극값을 뽑는다
function trace(action) {
  const h = fresh();
  for (let i = 0; i < 5; i++) step(h, 16);        // 중립 안정화
  h.setAction(action);
  const rec = { sx: [], sz: [], ex: [] };
  const right = !action.startsWith('LEFT');
  for (let i = 0; i < 32; i++) {
    step(h, 16);
    const arm = right ? h.armR : h.armL;
    rec.sx.push(arm.shoulder.rotation.x);
    rec.sz.push(arm.shoulder.rotation.z);
    rec.ex.push(arm.elbow.rotation.x);
  }
  const rng = a => Math.max(...a) - Math.min(...a);
  return {
    reachMin: Math.min(...rec.sx),          // 팔을 얼마나 앞/위로 올렸나 (작을수록 높이)
    swing:    rng(rec.sz),                  // 좌우 스윙 폭 → 훅의 서명
    extend:   Math.max(...rec.ex),          // 팔꿈치가 얼마나 펴졌나 (0에 가까울수록 곧음)
    elbowMin: Math.min(...rec.ex),
  };
}

const R = {};
['RIGHT_CROSS','RIGHT_HOOK','RIGHT_UPPERCUT','LEFT_JAB','LEFT_HOOK','LEFT_UPPERCUT'].forEach(a => R[a] = trace(a));

console.log('기술            어깨X최소  스윙폭   팔꿈치최대(펴짐)');
for (const [a, v] of Object.entries(R)) {
  console.log(`${a.padEnd(15)} ${v.reachMin.toFixed(2).padStart(7)} ${v.swing.toFixed(2).padStart(8)} ${v.extend.toFixed(2).padStart(12)}`);
}

let fail = 0;
const ck = (name, cond) => { console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}`); if (!cond) fail++; };
console.log('\n--- 기술이 서로 구별되는가 ---');
ck('스트레이트: 팔꿈치를 편다 (>-0.5)',        R.RIGHT_CROSS.extend > -0.5);
ck('훅: 팔꿈치를 접은 채 유지 (<-1.0)',        R.RIGHT_HOOK.extend < -1.0);
ck('훅: 좌우 스윙이 가장 크다',                R.RIGHT_HOOK.swing > R.RIGHT_CROSS.swing * 3 && R.RIGHT_HOOK.swing > R.RIGHT_UPPERCUT.swing * 3);
ck('어퍼컷: 팔을 가장 높이 올린다',            R.RIGHT_UPPERCUT.reachMin < R.RIGHT_CROSS.reachMin);
ck('어퍼컷: 팔꿈치를 깊게 접는다 (<-1.5)',     R.RIGHT_UPPERCUT.extend < -1.5);
ck('좌우 대칭: 훅 스윙폭이 좌우 동일',         Math.abs(R.RIGHT_HOOK.swing - R.LEFT_HOOK.swing) < 0.02);
ck('좌우 대칭: 어퍼 도달높이 좌우 동일',       Math.abs(R.RIGHT_UPPERCUT.reachMin - R.LEFT_UPPERCUT.reachMin) < 0.02);

console.log('\n--- 치지 않는 팔은 가드를 유지하는가 ---');
{
  const h = fresh();
  for (let i = 0; i < 5; i++) step(h, 16);
  h.setAction('RIGHT_CROSS');
  for (let i = 0; i < 12; i++) step(h, 16);
  ck('오른손 크로스 중 왼팔은 가드(팔꿈치 접힘 <-1.6)', h.armL.elbow.rotation.x < -1.6);
  ck('오른팔은 뻗어 있다 (팔꿈치 >-1.0)',                h.armR.elbow.rotation.x > -1.0);
}

console.log('\n--- K.O. 다운 (#5) ---');
{
  const h = fresh();
  for (let i = 0; i < 5; i++) step(h, 16);
  ck('평상시 보인다', h.group.visible === true);
  h.setDown(true);
  for (let i = 0; i < 70; i++) step(h, 16);   // 1.12초
  ck('쓰러진 뒤 링에서 사라진다', h.group.visible === false);
  ck('뒤로 눕는다 (rig.rotation.x < -1.0)', h.rig.rotation.x < -1.0);
  h.setDown(false);
  for (let i = 0; i < 45; i++) step(h, 16);
  ck('라운드 리셋으로 부활한다', h.group.visible === true && Math.abs(h.rig.rotation.x) < 0.1);
}

console.log('\n--- 피격 리액션 (#4) ---');
{
  const h = fresh();
  for (let i = 0; i < 5; i++) step(h, 16);
  const z0 = h.rig.position.z;
  h.hit(8);
  step(h, 16);
  ck('맞으면 뒤(-z)로 밀린다', h.rig.position.z < z0 - 0.1);
  ck('맞으면 상체가 젖혀진다', h.rig.rotation.x < -0.05);
  for (let i = 0; i < 30; i++) step(h, 16);
  ck('0.5초 뒤 원자세로 복귀', Math.abs(h.rig.position.z) < 0.02 && Math.abs(h.rig.rotation.x) < 0.02);
}

console.log('');
console.log('--- 3D 얼굴 배치: 두개골 구에 파묻히지 않는가 ---');
// 머리 반지름은 humanoid.js 에서 읽는다 — 비율을 바꿔도 테스트가 따라오도록
const HEAD_R_EXPECTED = Number(
  require('fs').readFileSync(require('path').join(__dirname, '../server/static/humanoid.js'), 'utf8')
    .match(/const HEAD_R = ([\d.]+)/)[1]);
{
  // 실제로 겪은 버그: 얼굴 메쉬를 구(반지름 1.3) 안쪽 z=0.62 에 두어
  // 얼굴 전체가 구에 파묻혀 host·1인칭 양쪽에서 아무것도 안 보였다.
  // "모든 얼굴 정점이 두개골 타원체 바깥에 있다"를 수치로 확인한다.

  // 얼굴 스텁 — face3d.js 의 createFace3D 반환 형태를 흉내낸다
  function fakeFace(bounds, verts) {
    const mesh = new THREE.Mesh(null, null);
    mesh.geometry = { attributes: { position: { array: verts } } };
    return {
      mesh, bounds,
      update() {}, hit() {}, setHp() {}, dispose() {},
    };
  }

  // 사람 얼굴 비슷한 정점 분포 (폭 2.6 기준). **속이 채워진 곡면**이어야 한다 —
  // 테두리만 있는 링으로 만들면 가운데(코·입 주변) 정점이 없어, 정작 가장 깊이 박히는
  // 영역을 검사하지 못한다. 실제 얼굴은 중앙이 두개골에 가장 가깝다.
  const verts = [];
  const RING = 14, SPOKE = 16;
  for (let r = 0; r <= RING; r++) {
    const t = r / RING;                       // 0=중앙, 1=테두리
    for (let k = 0; k < SPOKE; k++) {
      const a = (k / SPOKE) * Math.PI * 2;
      const rx = 1.28 * t * (0.62 + 0.38 * Math.abs(Math.cos(a)));
      const ry = 1.55 * t;
      // 중앙이 앞으로 튀어나온 곡면 (코 쪽이 +z, 가장자리는 뒤로 감김)
      const z = 0.62 * Math.cos(t * Math.PI * 0.72) - 0.30;
      verts.push(Math.cos(a) * rx, Math.sin(a) * ry, z);
    }
  }
  verts.push(0, -0.10, 0.78);       // 코끝
  const zs = [];
  for (let i = 2; i < verts.length; i += 3) zs.push(verts[i]);
  const bounds = { zMin: Math.min(...zs), zMax: Math.max(...zs),
                   xMin: -1.2, xMax: 1.2, yMin: -1.5, yMax: 1.5 };

  const h = window.createHumanoid(0xff3366);
  // 얼굴을 씌우기 "전"의 머리 모양을 기억해 둔다 — 복귀 검사의 기준값이다.
  // 1.0 을 기대값으로 박아 두면 기본 머리 비율을 바꿀 때마다 테스트가 깨진다.
  const headScale0 = [h.head.scale.x, h.head.scale.y, h.head.scale.z];
  h.setFace(fakeFace(bounds, verts));
  h.update();

  const faceZ = h.getFace().mesh.position.z;
  const fk = h.getFace().mesh.scale.x;          // 머리 크기에 맞춘 스케일
  const sx = h.head.scale.x, sy = h.head.scale.y, sz = h.head.scale.z;
  const R = HEAD_R_EXPECTED;

  // 두개골 타원체: (x/(R*sx))^2 + (y/(R*sy))^2 + (z/(R*sz))^2 = 1
  // 얼굴 정점을 머리 좌표계로 옮겨(z += faceZ) 전부 바깥(>1)인지 본다.
  let inside = 0, worst = Infinity;
  for (let i = 0; i < verts.length; i += 3) {
    const x = verts[i] * fk, y = verts[i + 1] * fk, z = verts[i + 2] * fk + faceZ;
    const e = (x / (R * sx)) ** 2 + (y / (R * sy)) ** 2 + (z / (R * sz)) ** 2;
    if (e < 1) inside++;
    if (e < worst) worst = e;
  }
  console.log('       얼굴 z 배치 ' + faceZ.toFixed(2)
            + ' · 두개골 스케일 (' + sx + ',' + sy + ',' + sz + ')'
            + ' · 가장 깊이 박힌 정점 ' + worst.toFixed(2) + ' (1보다 커야 바깥)');
  ck('모든 얼굴 정점이 두개골 바깥에 있다', inside === 0, inside + '개가 구 안에 박힘');
  ck('얼굴이 두개골 앞쪽(+z)에 놓인다', faceZ > 0, faceZ.toFixed(2));
  ck('두개골이 z로 납작하다 (뒤통수 역할)', sz < sx, 'z ' + sz + ' < x ' + sx);
  ck('바이저는 얼굴과 겹치므로 숨긴다', h.visor.visible === false);
  ck('두개골 구는 남는다 (떠 있는 가면 방지)', h.head.visible === true);

  // 얼굴을 떼면 원래 머리로 복귀
  h.setFace(null);
  h.update();
  ck('얼굴을 떼면 원래 머리로 복귀',
     Math.abs(h.head.scale.x - headScale0[0]) < 1e-6
     && Math.abs(h.head.scale.z - headScale0[2]) < 1e-6
     && h.visor.visible === true,
     `scale ${h.head.scale.x} vs ${headScale0[0]}`);

  // 깊이가 다른 얼굴(납작한 얼굴)도 파묻히지 않는가
  const flatBounds = Object.assign({}, bounds, { zMin: -0.12, zMax: 0.18 });
  const flatVerts = verts.slice();
  for (let i = 2; i < flatVerts.length; i += 3) flatVerts[i] *= 0.25;
  const h2 = window.createHumanoid(0x00e5ff);
  h2.setFace(fakeFace(flatBounds, flatVerts));
  h2.update();
  const fz2 = h2.getFace().mesh.position.z;
  const fk2 = h2.getFace().mesh.scale.x;
  let inside2 = 0;
  for (let i = 0; i < flatVerts.length; i += 3) {
    const x = flatVerts[i] * fk2, y = flatVerts[i + 1] * fk2, z = flatVerts[i + 2] * fk2 + fz2;
    if ((x / (R * h2.head.scale.x)) ** 2 + (y / (R * h2.head.scale.y)) ** 2
      + (z / (R * h2.head.scale.z)) ** 2 < 1) inside2++;
  }
  ck('납작한 얼굴도 파묻히지 않는다 (배치가 바운딩에서 역산된다)', inside2 === 0,
     inside2 + '개 박힘');
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
