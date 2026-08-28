/**
 * head_pixel_harness.js — 복원된 머리를 **실제로 렌더해 픽셀을 본다.**
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd iter3/tests && node head_pixel_harness.js [베이스URL]
 *
 * 왜 필요한가:
 *   3장 촬영을 넣었을 때 UV 검사(어느 칸을 무는가)는 전부 통과했는데 실제 화면에서는
 *   **천장과 벽이 뒤통수에 발려 있었다.** 두개골 정점을 옆사진에 투영했더니 머리 실루엣
 *   **바깥**으로 나가서 배경을 찍은 것이다. UV 가 "옆면 칸 안"인 것은 맞았다 —
 *   그 칸의 그 자리가 머리가 아니었을 뿐이다.
 *
 *   좌표를 세는 검사로는 이 부류를 절대 못 잡는다. 그래서 여기서는
 *   **배경을 형광색으로 칠한 합성 사진**으로 얼굴을 만들고, 머리를 여러 각도에서 렌더해
 *   그 형광색이 한 픽셀이라도 머리에 묻었는지 본다.
 */
const { open } = require('./_cdp');
const BASE = process.argv[2] || 'https://localhost:8000';

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

/**
 * 페이지 안에서 도는 실험.
 *
 * 합성 "사진" 3장을 만든다 — 배경은 **마젠타(255,0,255)**, 얼굴은 살구색, 머리 위는 갈색.
 * 실제 사진에서 천장·벽에 해당하는 자리를 마젠타로 칠해 두는 것이다.
 * 복원된 머리를 앞·옆·뒤에서 렌더해 마젠타가 보이면 = 배경을 텍스처로 쓴 것.
 */
const LAB = `(async () => {
  const N = 468;
  const SKIN = [230, 184, 156], HAIR = [64, 40, 28], BG = [255, 0, 255];

  // ── 합성 사진 한 장 만들기 ──────────────────────────────────────
  // yaw 만큼 돌린 얼굴을 그린다. 타원(얼굴) + 그 위 띠(머리카락), 나머지는 배경.
  function makePhoto(yaw) {
    const S = 256;
    const c = document.createElement('canvas');
    c.width = S; c.height = S;
    const g = c.getContext('2d');
    g.fillStyle = 'rgb(' + BG.join(',') + ')';
    g.fillRect(0, 0, S, S);
    const rx = S * 0.22 * Math.cos(yaw * 0.9) + S * 0.06;   // 옆으로 돌면 좁아진다
    const ry = S * 0.30;
    const cx = S * 0.5 - Math.sin(yaw) * S * 0.05, cy = S * 0.52;
    // 머리카락 (얼굴보다 크게, 위·뒤로)
    g.fillStyle = 'rgb(' + HAIR.join(',') + ')';
    g.beginPath();
    g.ellipse(cx - Math.sin(yaw) * S * 0.10, cy - S * 0.10, rx * 1.35, ry * 1.15, 0, 0, 7);
    g.fill();
    // 얼굴
    g.fillStyle = 'rgb(' + SKIN.join(',') + ')';
    g.beginPath();
    g.ellipse(cx, cy, rx, ry, 0, 0, 7);
    g.fill();
    return c;
  }

  // ── 랜드마크 ────────────────────────────────────────────────────
  const front = new Array(N);
  for (let i = 0; i < N; i++) {
    const a = i / N * Math.PI * 2, r = 0.16 + 0.05 * Math.sin(i * 2.7);
    front[i] = { x: 0.5 + Math.cos(a) * r * 0.78, y: 0.52 + Math.sin(a) * r, z: -0.05 * Math.cos(a) };
  }
  const OV = (typeof FACEMESH_FACE_OVAL !== 'undefined') ? FACEMESH_FACE_OVAL : null;
  if (OV) {
    const seen = new Set();
    OV.forEach(e => { seen.add(e[0]); seen.add(e[1]); });
    const ids = [...seen];
    ids.forEach((id, k) => {
      const a = k / ids.length * Math.PI * 2;
      front[id] = { x: 0.5 + Math.cos(a) * 0.20, y: 0.52 + Math.sin(a) * 0.28, z: 0.02 };
    });
  }
  function turned(ang) {
    const c = Math.cos(ang), s = Math.sin(ang);
    return front.map(q => {
      const x = q.x - 0.5, z = q.z;
      return { x: 0.5 + (x * c + z * s) - Math.sin(ang) * 0.05, y: q.y, z: (-x * s + z * c) };
    });
  }

  // ── 아틀라스 ────────────────────────────────────────────────────
  const AW = 1024, AH = 512;
  const at = document.createElement('canvas');
  at.width = AW; at.height = AH;
  const ag = at.getContext('2d');
  ag.fillStyle = '#000'; ag.fillRect(0, 0, AW, AH);
  ag.drawImage(makePhoto(0), 0, 0, 512, 512);
  ag.drawImage(makePhoto(-0.9), 512, 0, 256, 256);
  ag.drawImage(makePhoto(0.9), 512, 256, 256, 256);

  const atlas = {
    front:   { x: 0,   y: 0,   w: 0.5,  h: 1   },
    sideNeg: { x: 0.5, y: 0,   w: 0.25, h: 0.5 },
    sidePos: { x: 0.5, y: 0.5, w: 0.25, h: 0.5 },
  };
  const full = { x0: 0, y0: 0, w: 1, h: 1 };

  const img = new Image();
  await new Promise(r => { img.onload = r; img.onerror = r; img.src = at.toDataURL('image/png'); });

  const f = window.createFace3D({
    landmarks: front, uvLandmarks: front, image: img, width: 2.6, aspect: 1,
    crop: full, imageW: 1, imageH: 1, atlas,
    sideViews: {
      neg: { lm: turned(-0.9), crop: full, imageW: 1, imageH: 1 },
      pos: { lm: turned(0.9),  crop: full, imageW: 1, imageH: 1 },
    },
  });
  if (!f) return JSON.stringify({ err: 'createFace3D null' });

  // ── 각도별로 렌더해 픽셀을 센다 ──────────────────────────────────
  const SZ = 220;
  const rc = document.createElement('canvas');
  rc.width = SZ; rc.height = SZ;
  const rend = new THREE.WebGLRenderer({ canvas: rc, antialias: false, alpha: false });
  rend.setClearColor(0x102030, 1);                 // 배경은 어두운 남색 (마젠타와 확실히 구분)
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  const pivot = new THREE.Object3D();
  pivot.add(f.mesh);
  scene.add(pivot);

  const b = f.bounds;
  const cx = (b.xMin + b.xMax) / 2, cy = (b.yMin + b.yMax) / 2, cz = (b.zMin + b.zMax) / 2;
  // 머리 전체가 화면에 들어와야 한다. 너무 가까우면 뒤에서 봐도 옆면이 잔뜩 잡혀
  // "뒤통수 색"을 재는 의미가 없어진다. fov 35도 기준으로 여유 있게 물린다.
  const R = Math.max(b.xMax - b.xMin, b.yMax - b.yMin, b.zMax - b.zMin) * 2.8;
  const cam = new THREE.PerspectiveCamera(35, 1, 0.1, 100);

  const views = { front: 0, side: Math.PI / 2, back: Math.PI, side2: -Math.PI / 2 };
  const out = {};
  for (const [name, ang] of Object.entries(views)) {
    pivot.rotation.y = ang;
    cam.position.set(cx, cy, cz + R);
    cam.lookAt(cx, cy, cz);
    rend.render(scene, cam);
    const d = rc.getContext('2d') ? null : null;
    const px = new Uint8Array(SZ * SZ * 4);
    rend.getContext().readPixels(0, 0, SZ, SZ, 0x1908 /*RGBA*/, 0x1401 /*UNSIGNED_BYTE*/, px);
    let bgMagenta = 0, skin = 0, hair = 0, head = 0, sr = 0, sg = 0, sb = 0;
    for (let i = 0; i < px.length; i += 4) {
      const r = px[i], g2 = px[i+1], bl = px[i+2];
      // 렌더 배경(남색)은 제외
      if (r < 40 && g2 < 60 && bl > 40 && bl < 70) continue;
      head++;
      sr += r; sg += g2; sb += bl;
      // 사진 배경(마젠타)은 초록 성분이 유독 낮은 것으로 가린다 — 조명이 곱해져도 유지된다
      if (r > 90 && bl > 90 && g2 < Math.min(r, bl) * 0.55) { bgMagenta++; continue; }
      // 피부와 머리카락은 밝기 차가 크다(기준색 194 vs 46). 조명은 둘 다 같은 비율로
      // 스케일하므로 밝기로 가르는 편이 색상 범위로 가르는 것보다 튼튼하다.
      const lum = r * 0.299 + g2 * 0.587 + bl * 0.114;
      if (lum > 115) skin++;
      else if (lum < 90) hair++;
    }
    out[name] = { head, bgMagenta, skin, hair,
                  mean: head ? [Math.round(sr/head), Math.round(sg/head), Math.round(sb/head)] : null,
                  magentaPct: head ? +(bgMagenta / head * 100).toFixed(1) : 0 };
  }
  rend.dispose();
  return JSON.stringify(out);
})()`;

(async () => {
  console.log('--- 복원된 머리를 렌더해 픽셀로 검사 ---');
  const a = await open(`${BASE}/arena`, { port: 9381, settle: 8000, fakeMedia: false });
  const raw = await a.evaluate(LAB);
  let r;
  try { r = JSON.parse(raw); } catch (e) { console.log('  결과 파싱 실패:', raw); process.exit(1); }
  if (r.err) { console.log('  오류:', r.err); process.exit(1); }

  for (const [name, v] of Object.entries(r)) {
    console.log(`  [${name}] 머리 픽셀 ${v.head} · 배경색 ${v.bgMagenta} (${v.magentaPct}%)`
      + ` · 피부 ${v.skin} · 머리카락 ${v.hair} · 평균색 rgb(${(v.mean||[]).join(',')})`);
  }
  console.log('');

  // 뒤통수는 만들지 않는다 — face3d 는 **얼굴 앞면만** 내놓고, 머리 뒤쪽은
  // humanoid 의 구형 두개골이 맡는다. 그래서 여기서는 옆·뒤에서 면적을 기대하지 않는다.
  ck('얼굴이 실제로 그려진다 (앞)', r.front.head > 2000, `${r.front.head}px`);
  ck('옆에서도 얼굴 일부가 보인다', r.side.head > 500, `${r.side.head}px`);

  // 핵심 — 사진의 배경이 머리에 묻지 않았는가
  for (const name of ['front', 'side', 'back', 'side2']) {
    ck(`${name}: 사진 배경이 머리에 묻지 않았다`, r[name].magentaPct < 2.0,
       `${r[name].magentaPct}%`);
  }

  // 이 하니스의 본래 목적은 **사진 배경이 얼굴에 발리는지**를 픽셀로 잡는 것이다.
  // (실제로 천장·벽이 뒤통수에 발린 적이 있다 — UV 좌표 검사로는 못 잡았다.)
  ck('어느 각도에서도 배경색이 2% 미만',
     ['front','side','back','side2'].every(k => r[k].magentaPct < 2.0),
     ['front','side','back','side2'].map(k => `${k} ${r[k].magentaPct}%`).join(' · '));

  await a.close();
  console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
  process.exit(fail ? 1 : 0);
})();
