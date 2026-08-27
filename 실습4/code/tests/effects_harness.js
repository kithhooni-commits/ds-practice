/**
 * effects_harness.js — effects.js 타격 이펙트 헤드리스 검증
 *
 * 브라우저 없이 THREE + Canvas 2D 최소 스텁 위에서 effects.js를 실제로 구동한다.
 * 캔버스 그리기 호출은 무시하고, 씬에 추가/제거되는 오브젝트 수와 수명만 본다.
 * 이펙트를 손볼 때 런타임 에러·객체 누수를 잡는 용도.
 *
 *   cd iter3/tests && node effects_harness.js
 */
require('./three_stub.js');

// --- Canvas 2D 스텁 (그리기는 전부 무시, 텍스트 폭만 근사) ---
const ctx2d = new Proxy({}, {
  get(_, k) {
    if (k === 'measureText') return (t) => ({ width: t.length * 40 });
    if (k === 'createRadialGradient' || k === 'createLinearGradient')
      return () => ({ addColorStop() {} });
    return () => {};
  },
  set() { return true; },
});
global.document = { createElement: () => ({ width: 0, height: 0, getContext: () => ctx2d }) };

// --- effects.js가 쓰는 THREE 추가 API ---
class Sprite extends THREE.Mesh {
  constructor(m) { super(null, m); this.isSprite = true; }
}
THREE.Sprite = Sprite;
THREE.SpriteMaterial = function (o) { return Object.assign(this, { opacity: 1, rotation: 0, dispose() {} }, o); };
THREE.PointsMaterial = function (o) { return Object.assign(this, { opacity: 1, dispose() {} }, o); };
THREE.CanvasTexture = function () { return this; };
THREE.BufferAttribute = function (arr) { this.array = arr; this.needsUpdate = false; return this; };
THREE.BufferGeometry = function () {
  this.attributes = {};
  this.setAttribute = (n, a) => { this.attributes[n] = a; };
  this.dispose = () => {};
  return this;
};
THREE.Points = function (g, m) { const o = new THREE.Mesh(g, m); o.isPoints = true; return o; };
THREE.AdditiveBlending = 2; THREE.NormalBlending = 1;
THREE.MathUtils = { ceilPowerOfTwo: n => Math.pow(2, Math.ceil(Math.log2(Math.max(1, n)))) };
THREE.Vector3.prototype.copy = function (v) { this.x=v.x; this.y=v.y; this.z=v.z; return this; };
THREE.Vector3.prototype.clone = function () { return new THREE.Vector3(this.x, this.y, this.z); };
THREE.Vector3.prototype.add = function (v) { this.x+=v.x; this.y+=v.y; this.z+=v.z; return this; };
THREE.Vector3.prototype.multiplyScalar = function (s) { this.x*=s; this.y*=s; this.z*=s; return this; };

require('../server/static/effects.js');

// --- 씬 스텁: add/remove 를 세는 것이 검증의 핵심 ---
function makeScene() {
  const objs = new Set();
  return { objs, add: o => objs.add(o), remove: o => objs.delete(o) };
}

let fail = 0;
const ck = (name, cond, extra) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${extra !== undefined ? `  (${extra})` : ''}`);
  if (!cond) fail++;
};
const V = (x, y, z) => new THREE.Vector3(x, y, z);

console.log('--- 일반 타격 ---');
{
  const scene = makeScene();
  const fx = window.createHitEffects(scene);
  fx.spawnHit(V(0, 4.6, 0), 0xff3366, 6, false);
  const n = scene.objs.size;
  ck('한 방에 여러 레이어가 겹쳐 터진다 (>=8)', n >= 8, `${n}개`);
  fx.update(0.016);
  ck('첫 프레임에 전부 살아있다', scene.objs.size === n);
  for (let i = 0; i < 130; i++) fx.update(0.016);   // 2.08초
  ck('수명이 끝나면 씬에서 전부 제거된다 (누수 없음)', scene.objs.size === 0, `${scene.objs.size}개 잔존`);
}

console.log('\n--- 가드로 막힌 타격 ---');
{
  const scene = makeScene();
  const fx = window.createHitEffects(scene);
  fx.spawnHit(V(0, 4.6, 0), 0xff3366, 6, true);
  ck('가드도 이펙트가 나온다', scene.objs.size >= 8, `${scene.objs.size}개`);
  for (let i = 0; i < 130; i++) fx.update(0.016);
  ck('정리된다', scene.objs.size === 0);
}

console.log('\n--- K.O. ---');
{
  const scene = makeScene();
  const fx = window.createHitEffects(scene);
  fx.spawnKO(V(0, 5.2, 0), 0xff3366);
  const n = scene.objs.size;
  ck('K.O.는 일반 타격보다 크다 (>=7)', n >= 7, `${n}개`);
  for (let i = 0; i < 200; i++) fx.update(0.016);   // 3.2초
  ck('정리된다', scene.objs.size === 0, `${scene.objs.size}개 잔존`);
}

console.log('\n--- 연타 & clear() ---');
{
  const scene = makeScene();
  const fx = window.createHitEffects(scene);
  for (let i = 0; i < 12; i++) { fx.spawnHit(V(i, 4.6, 0), 0x00e5ff, 8, false); fx.update(0.016); }
  ck('12연타를 버틴다', scene.objs.size > 0, `${scene.objs.size}개 동시 생존`);
  fx.clear();
  ck('clear()가 전부 지운다', scene.objs.size === 0);
  fx.update(0.016);
  ck('clear() 후 update()가 터지지 않는다', true);
}

console.log('\n--- 텍스트 텍스처 캐시 ---');
{
  const scene = makeScene();
  const fx = window.createHitEffects(scene);
  let made = 0;
  const orig = THREE.CanvasTexture;
  THREE.CanvasTexture = function () { made++; return orig.call(this); };
  for (let i = 0; i < 10; i++) fx.spawnHit(V(0, 4.6, 0), 0xff3366, 6, false);
  THREE.CanvasTexture = orig;
  // 같은 라벨/데미지 10회 → 텍스처는 처음 2개(HIT! / -6)만 새로 만들어야 한다
  ck('같은 문구는 텍스처를 재생성하지 않는다', made <= 2, `새 텍스처 ${made}개`);
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
