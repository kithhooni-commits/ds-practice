/**
 * face_harness.js — face3d.js (웹캠 사진 → 3D 얼굴 복원 + 손상 표현) 헤드리스 검증
 *
 *   cd iter3/tests && node face_harness.js
 *
 * 브라우저 없이 THREE + Canvas 최소 스텁 위에서 face3d.js 를 실제로 구동한다.
 * 검증 대상은 눈으로 보기 어려운 것들이다:
 *   - FACEMESH_TESSELATION(간선 목록)에서 삼각형을 제대로 복원하는가
 *   - MediaPipe 의 비등방 정규화 좌표를 등방으로 되돌리는가 (안 하면 얼굴이 세로로 길어진다)
 *   - 맞은 자리가 실제로 눌렸다가 돌아오는가
 *   - HP 가 낮아지면 표정/코피가 나타나는가
 */
require('./three_stub.js');

// ── Canvas 2D 스텁 ────────────────────────────────────────────────────────
const ctx2d = new Proxy({}, {
  get(_, k) {
    if (k === 'createRadialGradient' || k === 'createLinearGradient') return () => ({ addColorStop() {} });
    if (k === 'measureText') return (t) => ({ width: t.length * 10 });
    return () => {};
  },
  set() { return true; },
});
global.document = {
  createElement: () => ({
    width: 0, height: 0,
    getContext: () => ctx2d,
    toDataURL: () => 'data:image/jpeg;base64,' + 'A'.repeat(2048),
  }),
};

// ── face3d.js 가 쓰는 THREE 추가 API ──────────────────────────────────────
THREE.MeshStandardMaterial = function (o) { return Object.assign(this, { opacity: 1, dispose() {} }, o); };
THREE.MeshBasicMaterial = function (o) { return Object.assign(this, { opacity: 1, dispose() {} }, o); };
THREE.CanvasTexture = function () { return Object.assign(this, { dispose() {} }); };
THREE.PlaneGeometry = function () { return Object.assign(this, { dispose() {} }); };
THREE.DoubleSide = 2;
THREE.BufferAttribute = function (arr, itemSize) {
  this.array = arr; this.itemSize = itemSize; this.needsUpdate = false; return this;
};
THREE.BufferGeometry = function () {
  this.attributes = {};
  this.index = null;
  this.setAttribute = (n, a) => { this.attributes[n] = a; };
  this.setIndex = (i) => { this.index = i; };
  this.dispose = () => {};
  // 부기(swell)가 법선 방향으로 부풀리므로 법선을 진짜로 계산해야 의미 있는 검증이 된다
  this.computeVertexNormals = () => {
    const pos = this.attributes.position.array;
    const n = pos.length / 3;
    let nrm = this.attributes.normal;
    if (!nrm) { nrm = new THREE.BufferAttribute(new Float32Array(pos.length), 3); this.attributes.normal = nrm; }
    const a = nrm.array; a.fill(0);
    const idx = this.index || [];
    for (let t = 0; t < idx.length; t += 3) {
      const i = idx[t] * 3, j = idx[t + 1] * 3, k = idx[t + 2] * 3;
      const ux = pos[j] - pos[i], uy = pos[j + 1] - pos[i + 1], uz = pos[j + 2] - pos[i + 2];
      const vx = pos[k] - pos[i], vy = pos[k + 1] - pos[i + 1], vz = pos[k + 2] - pos[i + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const b of [i, j, k]) { a[b] += nx; a[b + 1] += ny; a[b + 2] += nz; }
    }
    for (let i = 0; i < n; i++) {
      const l = Math.hypot(a[i * 3], a[i * 3 + 1], a[i * 3 + 2]) || 1;
      a[i * 3] /= l; a[i * 3 + 1] /= l; a[i * 3 + 2] /= l;
    }
  };
  return this;
};
THREE.Vector3.prototype.copy = function (v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; };

// 실제 CDN 데이터 대신, 알고리즘을 검증할 수 있는 합성 테셀레이션을 쓴다.
// (네트워크 의존 없이 돌아야 하므로. 토폴로지 규칙은 동일하다 — 간선 목록에서 삼각형 복원)
global.FACEMESH_TESSELATION = [];

require('../server/static/face3d.js');

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

// ── 1. 간선 → 삼각형 복원 ────────────────────────────────────────────────
console.log('--- 간선 목록에서 삼각형 복원 ---');
{
  const tri = window.__faceTrianglesFromEdges;
  // 삼각형 하나: 0-1-2
  ck('삼각형 1개', tri([[0, 1], [1, 2], [2, 0]]).length / 3 === 1);
  // 사각형을 두 삼각형으로: 0-1-2, 0-2-3
  const quad = tri([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]]);
  ck('사각형 → 삼각형 2개', quad.length / 3 === 2, `${quad.length / 3}개`);
  // 삼각형이 없는 열린 경로
  ck('닫히지 않은 경로는 삼각형 0개', tri([[0, 1], [1, 2], [2, 3]]).length === 0);
  // 같은 삼각형을 세 번 세지 않는지 (간선 순서를 뒤섞어도 1개)
  ck('중복 계수 없음', tri([[1, 0], [2, 1], [0, 2]]).length / 3 === 1);

  // 무작위 그래프에서 브루트포스(모든 3중 조합 검사)와 결과가 일치하는지 본다.
  // 격자 같은 "정답을 안다고 가정한" 케이스보다 훨씬 강한 검증이다 —
  // 인접 셀이 공유 간선으로 추가 삼각형을 만드는 것까지 자동으로 맞춰진다.
  function bruteForceTriangles(edges, n) {
    const has = new Set(edges.map(([a, b]) => (a < b ? a + ',' + b : b + ',' + a)));
    const e = (a, b) => has.has(a < b ? a + ',' + b : b + ',' + a);
    let count = 0;
    for (let a = 0; a < n; a++)
      for (let b = a + 1; b < n; b++)
        for (let c = b + 1; c < n; c++)
          if (e(a, b) && e(b, c) && e(a, c)) count++;
    return count;
  }
  let mismatch = 0, totalTris = 0;
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let trial = 0; trial < 20; trial++) {
    const n = 8 + Math.floor(rnd() * 8);
    const edges = [];
    for (let a = 0; a < n; a++)
      for (let b = a + 1; b < n; b++)
        if (rnd() < 0.45) edges.push([a, b]);
    const got = tri(edges).length / 3;
    const want = bruteForceTriangles(edges, n);
    totalTris += want;
    if (got !== want) mismatch++;
  }
  ck('무작위 그래프 20개에서 브루트포스와 완전 일치', mismatch === 0,
     `삼각형 총 ${totalTris}개, 불일치 ${mismatch}건`);

  // 실제 FACEMESH_TESSELATION 은 같은 간선을 [a,b] 와 [b,a] 로 **양방향 모두** 담는다.
  // 중복 제거를 빼먹으면 인접 리스트에 이웃이 두 번 들어가 같은 삼각형이 4번 잡힌다.
  // (실측으로 3288개 vs 822개였다.) 이 케이스가 그 회귀를 막는다.
  const oneWay = [[0, 1], [1, 2], [2, 0]];
  const twoWay = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]];
  ck('간선이 양방향으로 들어와도 삼각형은 1개',
     tri(twoWay).length / 3 === 1, `${tri(twoWay).length / 3}개`);
  ck('단방향/양방향 결과가 같다', tri(oneWay).length === tri(twoWay).length);

  // 사각형(삼각형 2개)도 양방향으로 넣어 확인
  const quad2 = [];
  [[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]].forEach(([a, b]) => quad2.push([a, b], [b, a]));
  ck('양방향 사각형도 삼각형 2개', tri(quad2).length / 3 === 2, `${tri(quad2).length / 3}개`);
}

// ── 테스트용 합성 얼굴 ────────────────────────────────────────────────────
// 468개 정점을 구면에 배치하고, MediaPipe 처럼 x는 폭으로 y는 높이로 나눈 좌표를 만든다.
const N = 468;
const ASPECT = 480 / 360;   // 4:3 웹캠

function makeSyntheticFace() {
  const lm = new Array(N);
  for (let i = 0; i < N; i++) {
    // 균등 구면 분포 (황금각 나선)
    const t = (i + 0.5) / N;
    const phi = Math.acos(1 - 2 * t);
    const th = Math.PI * (1 + Math.sqrt(5)) * i;
    const X = Math.sin(phi) * Math.cos(th) * 0.5;   // 실제(등방) 좌표, 반지름 0.5
    const Y = Math.sin(phi) * Math.sin(th) * 0.5;
    const Z = Math.cos(phi) * 0.5;
    // MediaPipe 정규화: x는 폭으로 나누므로 aspect 만큼 눌린다
    lm[i] = { x: 0.5 + X / ASPECT, y: 0.5 - Y, z: Z / ASPECT };
  }
  // 광대 인덱스는 스케일 기준이므로 좌우 대칭으로 명시 배치
  lm[234] = { x: 0.5 - 0.5 / ASPECT, y: 0.5, z: 0 };
  lm[454] = { x: 0.5 + 0.5 / ASPECT, y: 0.5, z: 0 };
  return lm;
}

/**
 * 얼굴 전체를 덮는 테셀레이션.
 * 실제 FACEMESH_TESSELATION 과 같은 형식으로 만든다 — 삼각형 (a,b,c) 마다
 * [a,b], [b,c], [c,a] 세 변을 **순서대로** 넣는다. getTriangles 의 주 경로가 이 형식을 읽는다.
 */
function fullTessellation(n) {
  const E = [];
  for (let i = 1; i < n - 1; i++) E.push([0, i], [i, i + 1], [i + 1, 0]);
  return E;
}

/** 3개씩 묶이지 않는(형식이 깨진) 테셀레이션 — 폴백 경로 확인용 */
function scrambledTessellation(n) {
  const E = [];
  for (let i = 1; i < n - 1; i++) E.push([0, i], [i + 1, 0], [i, i + 1]);   // 순서 뒤섞음
  return E;
}

console.log('');
console.log('--- 테셀레이션 읽기: 정확한 목록 우선, 형식이 깨지면 폴백 ---');
{
  const reload = () => {
    delete require.cache[require.resolve('../server/static/face3d.js')];
    require('../server/static/face3d.js');
  };
  const TRI_N = 40;

  // (1) 실제 형식 — 3개씩 묶인 목록을 그대로 읽는다
  global.FACEMESH_TESSELATION = fullTessellation(TRI_N);
  reload();
  const lmA = [];
  for (let i = 0; i < 468; i++) lmA.push({ x: 0.5 + i * 1e-4, y: 0.5 - i * 1e-4, z: i * 1e-4 });
  lmA[234] = { x: 0.3, y: 0.5, z: 0 };
  lmA[454] = { x: 0.7, y: 0.5, z: 0 };
  const fA = window.createFace3D({ landmarks: lmA, image: null, width: 2.6, aspect: 1 });
  ck('3개씩 묶인 목록을 그대로 읽는다', fA && fA.triangleCount === TRI_N - 2,
     `${fA ? fA.triangleCount : 0}개 (기대 ${TRI_N - 2})`);

  // (2) 형식이 깨졌을 때 — 3-사이클 폴백으로도 메쉬가 만들어진다
  global.FACEMESH_TESSELATION = scrambledTessellation(TRI_N);
  reload();
  const fB = window.createFace3D({ landmarks: lmA, image: null, width: 2.6, aspect: 1 });
  ck('형식이 깨지면 3-사이클로 폴백한다', fB && fB.triangleCount > 0,
     `${fB ? fB.triangleCount : 0}개`);
}

console.log('');
console.log('--- 종횡비 보정 (안 하면 얼굴이 세로로 길어진다) ---');
{
  global.FACEMESH_TESSELATION = fullTessellation(N);
  // 캐시를 비우기 위해 모듈을 다시 로드
  delete require.cache[require.resolve('../server/static/face3d.js')];
  require('../server/static/face3d.js');

  const lm = makeSyntheticFace();
  // 실제 공간에서 중심으로부터 같은 거리(0.5)인 두 점을 x축/y축에 놓는다
  lm[100] = { x: 0.5 + 0.5 / ASPECT, y: 0.5,       z: 0 };   // +x 로 0.5
  lm[200] = { x: 0.5,                y: 0.5 - 0.5, z: 0 };   // +y 로 0.5

  const withFix = window.createFace3D({ landmarks: lm, image: null, width: 2.6, aspect: ASPECT });
  const noFix   = window.createFace3D({ landmarks: lm, image: null, width: 2.6, aspect: 1 });

  const rad = (face, i) => {
    const a = face.mesh.geometry.attributes.position.array;
    return Math.hypot(a[i * 3], a[i * 3 + 1], a[i * 3 + 2]);
  };
  const rx = rad(withFix, 100), ry = rad(withFix, 200);
  const bx = rad(noFix, 100), by = rad(noFix, 200);

  ck('보정하면 x/y 축 반지름이 같다', Math.abs(rx - ry) / ry < 0.03,
     `x ${rx.toFixed(3)} vs y ${ry.toFixed(3)}`);
  ck('보정하지 않으면 실제로 어긋난다 (테스트가 유효함을 확인)',
     Math.abs(bx - by) / by > 0.15, `x ${bx.toFixed(3)} vs y ${by.toFixed(3)}`);
  ck('머리 폭이 지정한 크기에 맞는다', Math.abs(rad(withFix, 234) + rad(withFix, 454) - 2.6) < 0.15,
     `${(rad(withFix, 234) + rad(withFix, 454)).toFixed(2)} / 2.6`);
}

console.log('');
console.log('--- 피격: 맞은 자리가 눌렸다가 돌아온다 ---');
{
  const lm = makeSyntheticFace();
  const face = window.createFace3D({ landmarks: lm, image: null, width: 2.6, aspect: ASPECT });
  const pos = face.mesh.geometry.attributes.position.array;
  const CHEEK_L = 234;
  const z0 = pos[CHEEK_L * 3 + 2];

  face.hit(8, 'left');
  let minZ = Infinity;
  for (let i = 0; i < 20; i++) { face.update(1 / 60); minZ = Math.min(minZ, pos[CHEEK_L * 3 + 2]); }
  ck('맞은 지점이 안쪽으로 눌린다', minZ < z0 - 0.02, `${z0.toFixed(3)} → ${minZ.toFixed(3)}`);

  for (let i = 0; i < 80; i++) face.update(1 / 60);
  ck('시간이 지나면 원래 형태로 복귀', Math.abs(pos[CHEEK_L * 3 + 2] - z0) < 0.02,
     `${pos[CHEEK_L * 3 + 2].toFixed(3)} vs ${z0.toFixed(3)}`);

  // 반대쪽 뺨은 영향을 받지 않아야 한다 (국소 변형)
  const CHEEK_R = 454;
  const zr0 = pos[CHEEK_R * 3 + 2];
  face.hit(8, 'left');
  let maxDev = 0;
  for (let i = 0; i < 20; i++) { face.update(1 / 60); maxDev = Math.max(maxDev, Math.abs(pos[CHEEK_R * 3 + 2] - zr0)); }
  ck('반대쪽 뺨은 거의 안 움직인다 (국소 변형)', maxDev < 0.02, `편차 ${maxDev.toFixed(4)}`);

  // 멍 — 정점 색이 어두워지고 누적된다
  const col = face.mesh.geometry.attributes.color.array;
  const g0 = col[CHEEK_L * 3 + 1];
  face.hit(8, 'left');
  const g1 = col[CHEEK_L * 3 + 1];
  ck('맞은 자리에 멍이 든다 (정점 색 어두워짐)', g1 < g0, `${g0.toFixed(2)} → ${g1.toFixed(2)}`);
  face.hit(8, 'left');
  ck('멍은 누적된다', col[CHEEK_L * 3 + 1] < g1);
  ck('붉은 기가 남는다 (R > G)', col[CHEEK_L * 3] > col[CHEEK_L * 3 + 1]);
}

console.log('');
console.log('--- HP: 낮을수록 지친 표정 + 코피 ---');
{
  const lm = makeSyntheticFace();
  const face = window.createFace3D({ landmarks: lm, image: null, width: 2.6, aspect: ASPECT });
  const pos = face.mesh.geometry.attributes.position.array;
  const CHIN = 152, LIP_LOWER = 14, LID_L = 159;

  face.setHp(100);
  for (let i = 0; i < 90; i++) face.update(1 / 60);
  const healthy = { chin: pos[CHIN * 3 + 1], lip: pos[LIP_LOWER * 3 + 1], lid: pos[LID_L * 3 + 1] };
  ck('HP 100 이면 코피 없음', face.state.bloodAmt === 0);

  face.setHp(10);
  for (let i = 0; i < 120; i++) face.update(1 / 60);
  ck('입이 벌어진다 (턱이 내려감)', pos[CHIN * 3 + 1] < healthy.chin - 0.02,
     `${healthy.chin.toFixed(3)} → ${pos[CHIN * 3 + 1].toFixed(3)}`);
  ck('아랫입술이 내려간다', pos[LIP_LOWER * 3 + 1] < healthy.lip - 0.01);
  ck('윗눈꺼풀이 처진다 (눈이 반쯤 감김)', pos[LID_L * 3 + 1] < healthy.lid - 0.01);
  ck('코피가 난다', face.state.bloodAmt > 0.8, face.state.bloodAmt.toFixed(2));

  face.setHp(100);
  for (let i = 0; i < 180; i++) face.update(1 / 60);
  ck('회복하면 표정도 돌아온다', Math.abs(pos[CHIN * 3 + 1] - healthy.chin) < 0.02);
  ck('코피도 멎는다', face.state.bloodAmt === 0);

  // 코피 시작 구간
  face.setHp(70); ck('HP 70 은 아직 코피 없음', face.state.bloodAmt === 0);
  face.setHp(55); ck('HP 55 부터 조금씩', face.state.bloodAmt > 0 && face.state.bloodAmt < 0.3,
                     face.state.bloodAmt.toFixed(2));
}

console.log('');
console.log('--- 직렬화 (네트워크 전송) ---');
{
  const lm = makeSyntheticFace();
  const canvas = document.createElement('canvas');
  const blob = window.serializeFace(lm, canvas, { aspect: ASPECT, imageW: 480, imageH: 360,
                                                  crop: { x0: 10, y0: 5, w: 200, h: 200 } }, 0.75);
  ck('랜드마크 468개가 평탄화된다', blob.lm.length === N * 3, `${blob.lm.length}개 값`);
  ck('좌표는 소수 4자리로 자른다', blob.lm.every(v => Math.abs(v * 10000 - Math.round(v * 10000)) < 1e-6));
  ck('종횡비가 함께 실린다', blob.aspect === ASPECT);
  ck('crop 정보가 함께 실린다', blob.crop && blob.crop.w === 200);
  ck('텍스처가 data URL', /^data:image\/jpeg;base64,/.test(blob.tex));
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
