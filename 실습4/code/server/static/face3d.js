/**
 * face3d.js — 웹캠 사진 한 장에서 복원한 3D 얼굴 + 피격 손상 표현
 * (Three.js r128 전역 THREE 의존, ES module 아님)
 *
 * ── 왜 이 방식인가 ────────────────────────────────────────────────────────────
 * 기존 검토(DEVLOG 2026-08-24)는 세 가지를 놓고 저울질했다.
 *   A) 사진을 구(sphere)에 텍스처로 — 쉽지만 정면만 맞는 2.5D
 *   B) DECA/FLAME 오프라인 3D 복원 — 진짜 3D지만 Python·GPU·GLB로 시간 리스크 큼
 *   C) 다중 뷰 빌보드 하이브리드 — 단순성 대비 이득이 작음
 *
 * 네 번째 길이 있다. **MediaPipe Face Mesh 가 브라우저에서 468개 3D 랜드마크를 준다.**
 * 정해진 토폴로지(canonical topology)를 쓰므로 랜드마크를 그대로 정점으로 삼고
 * 테셀레이션으로 삼각형을 구성하면, 사진 한 장에서 **그 사람의 실제 3D 얼굴 메쉬**가 나온다.
 * B의 결과물(단안 3D 복원)을 A의 비용으로 얻는 셈이다.
 *
 * 그리고 이 선택이 게임에 직접 값을 한다 — 진짜 메쉬이므로
 * **맞으면 그 지점이 눌리고, HP가 떨어지면 표정이 바뀌는** 것을 정점 변형으로 표현할 수 있다.
 * 텍스처를 입힌 구였다면 불가능하다.
 *
 * FPS: Face Mesh 는 **캡처 순간에만** 돌리고 즉시 끈다. 게임 중에는 Pose 만 돈다.
 * (과거 두 모델 동시 실행 시 30fps 이하로 떨어진 이력이 있어 처음부터 배제한 설계다.)
 *
 * ── 사용법 ──────────────────────────────────────────────────────────────────
 *   const face = window.createFace3D({ landmarks, image, color });
 *   humanoid.setFace(face.mesh);
 *   face.hit(damage, 'left');   // 피격 — 그 방향이 눌리고 멍이 든다
 *   face.setHp(hp);             // 0~100 — 낮을수록 지친 표정 + 코피
 *   face.update(dt);            // 매 프레임
 *
 * 직렬화(네트워크 전송):
 *   const blob = window.serializeFace(landmarks, canvas);   // { lm, tex }
 *   const face = window.createFace3DFromBlob(blob, color);  // 수신 측
 */
(function () {
  if (typeof THREE === 'undefined') {
    console.error('[face3d.js] THREE 로드 필요');
    return;
  }

  // ── MediaPipe canonical face 랜드마크 인덱스 ──────────────────────────────
  // 표정을 만들 때 건드릴 부위. 인덱스는 Face Mesh 의 고정 토폴로지라 사람마다 같다.
  const IDX = {
    NOSE_TIP: 1,
    NOSE_BOTTOM: 2,
    NOSTRIL_L: 98, NOSTRIL_R: 327,
    BROW_L: [70, 63, 105, 66, 107],
    BROW_R: [336, 296, 334, 293, 300],
    LID_UPPER_L: [159, 158, 157, 173],
    LID_UPPER_R: [386, 385, 384, 398],
    LID_LOWER_L: [145, 144, 163],
    LID_LOWER_R: [374, 380, 381],
    LIP_UPPER: [13, 12, 82, 312],
    LIP_LOWER: [14, 15, 87, 317],
    MOUTH_CORNER_L: 61, MOUTH_CORNER_R: 291,
    CHIN: 152,
    CHEEK_L: 234, CHEEK_R: 454,
  };

  // 삼각형 목록은 토폴로지가 고정이라 한 번만 계산해 모든 얼굴이 공유한다.
  let cachedTris = null;

  /**
   * FACEMESH_TESSELATION 은 "삼각형"이 아니라 **간선 쌍** 목록이다.
   * 메쉬를 만들려면 삼각형이 필요하므로 3-사이클을 찾아 복원한다.
   * a<b<c 순서로만 담아 같은 삼각형이 세 번 잡히는 것을 막는다.
   */
  function trianglesFromEdges(edges) {
    const adj = new Map();
    const eset = new Set();
    const key = (a, b) => (a < b ? a * 100000 + b : b * 100000 + a);

    // 먼저 간선을 중복 제거한다. FACEMESH_TESSELATION 은 같은 간선을 [a,b] 와 [b,a] 로
    // **양방향 모두** 담고 있어(2556개 = 1278개 x 2), 그대로 인접 리스트를 만들면
    // 이웃이 두 번씩 들어가 같은 삼각형이 2x2 = 4번 잡힌다.
    // (실측: 중복 제거 전 3288개 → 제거 후 822개. 위상적으로 기대되는 값은 약 811개다.)
    for (const e of edges) {
      const a = e[0], b = e[1];
      if (a === b) continue;
      eset.add(key(a, b));
    }
    for (const k of eset) {
      const a = Math.floor(k / 100000), b = k % 100000;
      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);
      adj.get(a).push(b);
      adj.get(b).push(a);
    }

    const tris = [];
    for (const [a, nbrs] of adj) {
      for (const b of nbrs) {
        if (b <= a) continue;
        for (const c of adj.get(b)) {
          if (c <= b) continue;
          if (eset.has(key(a, c))) tris.push(a, b, c);
        }
      }
    }
    return tris;
  }

  /**
   * FACEMESH_TESSELATION 에서 삼각형 목록을 얻는다.
   *
   * 이 배열은 **삼각형 목록에서 생성된 것**이라 3개씩 묶으면 한 삼각형의 세 변이다:
   *   [a,b], [b,c], [c,a]  →  삼각형 (a,b,c)
   * 실제로 확인했다 — 852개 그룹 전부가 닫힌 삼각형이고 어긋나는 그룹은 0개다.
   *
   * 3-사이클 탐색(trianglesFromEdges)으로도 852개를 전부 찾지만 **면이 아닌 가짜 삼각형 2개**를
   * 더 만들어낸다(854개). 그래프에는 존재하지만 메쉬의 면은 아닌 3-사이클이다.
   * 그 2개가 얼굴 위에 엉뚱한 폴리곤으로 얹혀 렌더가 지저분해진다.
   * 정확한 목록을 그대로 읽을 수 있으므로 추측할 이유가 없다.
   */
  function getTriangles() {
    if (cachedTris) return cachedTris;
    const tess = (typeof FACEMESH_TESSELATION !== 'undefined') ? FACEMESH_TESSELATION
               : (window.FACEMESH_TESSELATION || null);
    if (!tess) {
      console.error('[face3d.js] FACEMESH_TESSELATION 없음 — face_mesh.js 를 먼저 로드하세요');
      return null;
    }

    const tris = [];
    let grouped = true;
    for (let i = 0; i + 2 < tess.length; i += 3) {
      const e0 = tess[i], e1 = tess[i + 1], e2 = tess[i + 2];
      if (e0[1] !== e1[0] || e1[1] !== e2[0] || e2[1] !== e0[0]) { grouped = false; break; }
      tris.push(e0[0], e1[0], e2[0]);
    }
    // 형식이 예상과 다르면(라이브러리 버전 차이 등) 3-사이클 탐색으로 폴백한다
    cachedTris = grouped ? tris : trianglesFromEdges(tess);
    if (!grouped) console.warn('[face3d.js] 테셀레이션이 3개씩 묶이지 않음 — 3-사이클 탐색으로 폴백');
    return cachedTris;
  }

  /**
   * 메시의 **구멍**을 찾는다.
   *
   * FACEMESH_TESSELATION 은 눈꺼풀·입술의 *둘레*만 덮고 **그 안쪽은 비워 둔다.**
   * 실측하면 852개 삼각형 중 눈 윤곽 정점만으로 이뤄진 삼각형이 **0개**다.
   * 그래서 눈·입 자리에는 그릴 면이 아예 없고, 뚫린 구멍으로 머리 안쪽이 들여다보인다.
   * "눈이 사라진다"의 정체가 이것이다 — 텍스처에 눈이 찍혀 있어도 붙일 면이 없다.
   *
   * 경계 간선(삼각형 **하나에만** 속하는 간선)을 이어 붙이면 닫힌 루프가 나온다.
   * 실측 결과 4개다 — 얼굴 바깥 테두리(36점) · 입(20점) · 양눈(16점씩).
   * 가장 큰 루프는 얼굴 바깥선이므로 제외하고 나머지를 구멍으로 본다.
   *
   * 인덱스를 하드코딩하지 않는다. 토폴로지가 바뀌어도(라이브러리 버전 차이) 따라간다.
   *
   * @param {number[]} tris 삼각형 인덱스 배열 (3개씩)
   * @returns {number[][]} 구멍 루프들 (각각 순서 있는 정점 인덱스)
   */
  /**
   * 3D 얼굴 좌표 -> 어떤 view 의 사진 좌표로 가는 **아핀 투영**을 최소자승으로 구한다.
   *
   * Face Mesh 는 어느 각도에서든 같은 468개 정점을 돌려주므로 대응점이 이미 짝지어져 있다.
   * 그래서 정합(registration) 없이 바로 풀 수 있다 — 같은 번호가 같은 지점이다.
   *
   * 구하는 것: u = a0*x + a1*y + a2*z + a3,  v = b0*x + b1*y + b2*z + b3
   * 미지수 8개에 대응점 468개이므로 과결정이고, 정규방정식 4x4 를 두 번 풀면 된다.
   *
   * 이걸 쓰면 두개골 정점(사진에 랜드마크가 없는 자리)도 **옆사진의 어디를 봐야 하는지**
   * 계산할 수 있다. 관자놀이·귀·귀 뒤 머리카락의 실제 픽셀이 거기 있다.
   *
   * @param {Float32Array} pos3  기준 3D 좌표 (x,y,z 반복)
   * @param {Array} lm2          그 view 의 랜드마크 (정규화 0..1)
   * @param {number} n           대응점 수
   * @returns {{a:number[], b:number[]}|null}
   */
  function fitProjection(pos3, lm2, n) {
    if (!lm2 || lm2.length < n) return null;
    // 정규방정식 M(4x4) * c = r(4)  — u 와 v 가 M 을 공유한다
    const M = new Float64Array(16);
    const ru = new Float64Array(4), rv = new Float64Array(4);
    for (let i = 0; i < n; i++) {
      const q = [pos3[i * 3], pos3[i * 3 + 1], pos3[i * 3 + 2], 1];
      const u = lm2[i].x, v = lm2[i].y;
      for (let a = 0; a < 4; a++) {
        for (let b = 0; b < 4; b++) M[a * 4 + b] += q[a] * q[b];
        ru[a] += q[a] * u;
        rv[a] += q[a] * v;
      }
    }
    // 정규방정식이 거의 특이해질 수 있다(정면 view 는 z 분산이 작다) — 약한 릿지를 넣는다
    for (let a = 0; a < 4; a++) M[a * 4 + a] += 1e-6 * (M[0] + M[5] + M[10] + M[15]) / 4;

    const solve = (rhs) => {
      const A = Array.from(M);
      const x = Array.from(rhs);
      for (let c = 0; c < 4; c++) {
        let piv = c;
        for (let r = c + 1; r < 4; r++) if (Math.abs(A[r * 4 + c]) > Math.abs(A[piv * 4 + c])) piv = r;
        if (Math.abs(A[piv * 4 + c]) < 1e-12) return null;
        if (piv !== c) {
          for (let k = 0; k < 4; k++) { const t = A[c * 4 + k]; A[c * 4 + k] = A[piv * 4 + k]; A[piv * 4 + k] = t; }
          const t = x[c]; x[c] = x[piv]; x[piv] = t;
        }
        for (let r = c + 1; r < 4; r++) {
          const f = A[r * 4 + c] / A[c * 4 + c];
          if (!f) continue;
          for (let k = c; k < 4; k++) A[r * 4 + k] -= f * A[c * 4 + k];
          x[r] -= f * x[c];
        }
      }
      for (let r = 3; r >= 0; r--) {
        let sum = x[r];
        for (let k = r + 1; k < 4; k++) sum -= A[r * 4 + k] * x[k];
        x[r] = sum / A[r * 4 + r];
      }
      return x;
    };

    const a = solve(ru), b = solve(rv);
    return (a && b) ? { a, b } : null;
  }

  function findHoles(tris) {
    const count = new Map();
    const key = (a, b) => (a < b ? a + '_' + b : b + '_' + a);
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t], b = tris[t + 1], c = tris[t + 2];
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const k = key(p, q);
        count.set(k, (count.get(k) || 0) + 1);
      }
    }
    // 경계 간선만 남긴다
    const adj = new Map();
    for (const [k, c] of count) {
      if (c !== 1) continue;
      const [a, b] = k.split('_').map(Number);
      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);
      adj.get(a).push(b);
      adj.get(b).push(a);
    }
    // 이어 붙여 루프로
    const seen = new Set(), loops = [];
    for (const start of adj.keys()) {
      if (seen.has(start)) continue;
      const loop = [start];
      seen.add(start);
      let cur = start;
      for (;;) {
        const nxt = (adj.get(cur) || []).find(x => !seen.has(x));
        if (nxt === undefined) break;
        seen.add(nxt);
        loop.push(nxt);
        cur = nxt;
      }
      if (loop.length >= 3) loops.push(loop);
    }
    if (!loops.length) return [];
    // 가장 큰 루프 = 얼굴 바깥 테두리. 이건 뒤통수를 붙일 자리이므로 메우면 안 된다.
    loops.sort((a, b) => b.length - a.length);
    return loops.slice(1);
  }

  /**
   * FACEMESH_FACE_OVAL 에서 **순서 있는 테두리 루프**를 뽑는다.
   * 간선 목록은 [a,b] 쌍이고 닫힌 고리를 이룬다(36개). 이어 붙여 정점 순서를 복원한다.
   * 이 고리가 "얼굴이 끝나는 선"이고, 여기서 뒤통수를 만들어 붙인다.
   */
  let cachedOval = null;
  function getFaceOval() {
    if (cachedOval) return cachedOval;
    const E = (typeof FACEMESH_FACE_OVAL !== 'undefined') ? FACEMESH_FACE_OVAL
            : (window.FACEMESH_FACE_OVAL || null);
    if (!E || !E.length) return null;
    const next = new Map();
    for (const e of E) next.set(e[0], e[1]);
    const start = E[0][0];
    const loop = [start];
    let cur = start;
    for (let i = 0; i < E.length + 4; i++) {
      cur = next.get(cur);
      if (cur === undefined || cur === start) break;
      loop.push(cur);
    }
    cachedOval = (loop.length >= 8) ? loop : null;
    return cachedOval;
  }

  /**
   * 랜드마크를 머리 크기에 맞게 정규화한다 (원점=얼굴 중심).
   *
   * **종횡비 보정이 핵심이다.** MediaPipe 정규화 좌표는 x를 이미지 *폭*으로,
   * y를 이미지 *높이*로 각각 나눈 값이라 등방(isotropic)이 아니다.
   * 480x360 프레임에서 그대로 쓰면 x가 0.75배로 눌려 얼굴이 세로로 길쭉해진다.
   * x에 aspect(W/H)를 곱해 실제 비율로 되돌린 뒤에 스케일을 잡아야 한다.
   * z는 MediaPipe 가 이미 "폭 기준"으로 낸 값이므로 x와 같은 단위다.
   */
  function normalizeLandmarks(lm, targetWidth, aspect) {
    const n = lm.length;
    const ar = aspect || 1;
    const pos = new Float32Array(n * 3);
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { cx += lm[i].x * ar; cy += lm[i].y; cz += lm[i].z * ar; }
    cx /= n; cy /= n; cz /= n;

    // 좌우 광대(234·454) 폭을 스케일 기준으로 삼는다 — 카메라 거리에 무관하게 일정하다
    const a = lm[IDX.CHEEK_L], b = lm[IDX.CHEEK_R];
    const span = Math.max(Math.hypot((a.x - b.x) * ar, a.y - b.y, (a.z - b.z) * ar), 1e-4);
    const s = targetWidth / span;

    for (let i = 0; i < n; i++) {
      // MediaPipe 는 y가 아래로 증가하고 z가 카메라 쪽으로 음수 → Three.js 좌표로 뒤집는다
      pos[i * 3]     = (lm[i].x * ar - cx) * s;
      pos[i * 3 + 1] = -(lm[i].y - cy) * s;
      pos[i * 3 + 2] = -(lm[i].z * ar - cz) * s;
    }
    return pos;
  }

  /**
   * 삼각형 winding 이 뒤집혔는지 판정해 필요하면 뒤집는다.
   * 코끝은 얼굴에서 가장 앞으로(+z) 튀어나온 점이므로, 코 주변 삼각형의 법선이
   * +z 를 향하지 않으면 전체가 뒤집힌 것이다.
   */
  function fixWinding(pos, tris) {
    let acc = 0;
    for (let t = 0; t < tris.length; t += 3) {
      const [i, j, k] = [tris[t], tris[t + 1], tris[t + 2]];
      const ax = pos[i * 3], ay = pos[i * 3 + 1], az = pos[i * 3 + 2];
      const bx = pos[j * 3], by = pos[j * 3 + 1], bz = pos[j * 3 + 2];
      const cx = pos[k * 3], cy = pos[k * 3 + 1], cz = pos[k * 3 + 2];
      const ux = bx - ax, uy = by - ay, uz = bz - az;
      const vx = cx - ax, vy = cy - ay, vz = cz - az;
      acc += ux * vy - uy * vx;          // 법선 z 성분
    }
    if (acc < 0) {
      for (let t = 0; t < tris.length; t += 3) {
        const tmp = tris[t + 1]; tris[t + 1] = tris[t + 2]; tris[t + 2] = tmp;
      }
    }
    return tris;
  }

  // ── 공용 텍스처 (코피 방울) ──────────────────────────────────────────────
  let bloodTex = null;
  function makeBloodTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(32, 26, 2, 32, 32, 30);
    grad.addColorStop(0.0, 'rgba(255,60,60,1)');
    grad.addColorStop(0.45, 'rgba(190,10,20,0.95)');
    grad.addColorStop(1.0, 'rgba(120,0,10,0)');
    g.fillStyle = grad;
    g.beginPath(); g.ellipse(32, 34, 16, 26, 0, 0, Math.PI * 2); g.fill();
    return new THREE.CanvasTexture(c);
  }

  /**
   * 얼굴 텍스처에서 평균 피부톤을 뽑는다.
   *
   * 얼굴 메쉬만 씌우면 **두개골 구는 여전히 원래 색**이라, 얼굴 가장자리에서 색이 뚝 끊긴다.
   * 그 경계선 때문에 "얼굴을 붙인 머리"가 아니라 "가면을 쓴 인형"으로 보인다.
   * 머리·목·팔다리를 이 톤으로 맞추면 경계가 사라진다.
   *
   * 가운데(볼·코 주변)만 표본으로 쓴다 — 가장자리는 머리카락·배경이 섞여 톤을 흐린다.
   * 너무 어둡거나(그림자) 너무 밝은(하이라이트) 픽셀도 뺀다.
   */
  /**
   * 사진에서 **머리 바깥을 머리카락색으로 덮는다.**
   *
   * 두개골 정점은 사진에 랜드마크가 없어 아핀 투영으로 위치를 추정하는데, 얼굴로 맞춘
   * 투영을 머리 뒤로 외삽하면 얼마든지 멀리 날아간다. 실제로 **천장과 벽이 뒤통수에 발렸다.**
   *
   * 투영을 정교하게 만드는 대신 **사진 쪽을 고친다** — 머리 실루엣 밖을 미리 머리카락색으로
   * 칠해 두면 어디를 찍든 배경이 나올 수 없다. UV 를 보간하다 그 위를 지나가도 마찬가지다.
   * "없는 데이터를 정확히 찍으려" 하는 대신 "없는 자리는 무난한 색"으로 만드는 쪽이 튼튼하다.
   *
   * @param {Image} image     아틀라스(또는 단일 사진)
   * @param {object} regions  { key: {x,y,w,h} } 정규화된 칸. null 이면 전체를 한 칸으로 본다.
   * @param {object} lmByKey  칸별 랜드마크 (그 칸에서 머리가 어디인지 재는 데 쓴다)
   * @returns {{canvas:HTMLCanvasElement, hair:number}|null}
   */
  function paintOutsideHead(image, regions, lmByKey, expand) {
    try {
      const W = image.naturalWidth || image.width;
      const H = image.naturalHeight || image.height;
      if (!W || !H) return null;
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      const g = c.getContext('2d');
      g.drawImage(image, 0, 0);

      // 머리카락 색 — 이마 바로 위 띠에서 딴다. 헤어라인 바로 위라 대부분 머리카락이고,
      // 머리가 없는 사람이면 이마 피부색이 나오는데 그것도 맞는 답이다.
      const fr = regions.front, flm = lmByKey.front;
      let hair = 0x3a2a20;
      if (fr && flm) {
        let mu = 0, mv = 0, ru = 0, rv = 0;
        for (let i = 0; i < flm.length; i++) { mu += flm[i].x; mv += flm[i].y; }
        mu /= flm.length; mv /= flm.length;
        for (let i = 0; i < flm.length; i++) {
          ru = Math.max(ru, Math.abs(flm[i].x - mu));
          rv = Math.max(rv, Math.abs(flm[i].y - mv));
        }
        // 이마 위쪽(정규화 v 가 작은 쪽) 띠
        const sx = (fr.x + (mu - ru * 0.55) * fr.w) * W;
        const sw = Math.max(2, ru * 1.10 * fr.w * W);
        // 랜드마크의 맨 위(= 헤어라인 근처)보다 **확실히 위**를 본다.
        // rv 는 중심에서 가장 먼 랜드마크까지의 거리이므로 mv - rv 가 곧 얼굴 꼭대기다.
        // 1.10~1.45 배 구간이면 얼굴을 벗어나 머리카락에 놓인다 — 얼굴에 걸치면
        // 머리카락색으로 피부색을 뽑게 되고, 그러면 뒤통수가 살색이 된다(실제로 그랬다).
        const sy = (fr.y + Math.max(0, mv - rv * 1.45) * fr.h) * H;
        const sh = Math.max(2, rv * 0.35 * fr.h * H);
        const d = g.getImageData(Math.max(0, sx | 0), Math.max(0, sy | 0),
                                 Math.min(W - (sx | 0), sw | 0), Math.min(H - (sy | 0), sh | 0)).data;
        let r = 0, gg = 0, b = 0, n = 0;
        for (let i = 0; i < d.length; i += 4) { r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++; }
        if (n > 4) hair = ((Math.round(r / n) << 16) | (Math.round(gg / n) << 8) | Math.round(b / n));
      }
      const hs = '#' + ('000000' + hair.toString(16)).slice(-6);

      // 칸마다 머리 타원 **밖**을 머리카락색으로 채운다
      for (const key of Object.keys(regions)) {
        const R = regions[key], L = lmByKey[key];
        if (!R || !L) continue;
        let mu = 0, mv = 0, ru = 0, rv = 0;
        for (let i = 0; i < L.length; i++) { mu += L[i].x; mv += L[i].y; }
        mu /= L.length; mv /= L.length;
        for (let i = 0; i < L.length; i++) {
          ru = Math.max(ru, Math.abs(L[i].x - mu));
          rv = Math.max(rv, Math.abs(L[i].y - mv));
        }
        const ex = expand || 1.30;
        g.save();
        g.beginPath();
        g.rect(R.x * W, R.y * H, R.w * W, R.h * H);
        g.ellipse((R.x + mu * R.w) * W, (R.y + mv * R.h) * H,
                  ru * ex * R.w * W, rv * ex * R.h * H, 0, 0, Math.PI * 2);
        g.clip('evenodd');
        g.fillStyle = hs;
        g.fillRect(R.x * W, R.y * H, R.w * W, R.h * H);
        g.restore();
      }
      return { canvas: c, hair };
    } catch (e) {
      return null;      // 다른 출처 이미지 등으로 캔버스가 오염되면 읽을 수 없다
    }
  }

  function sampleSkinTone(image, frontRegion) {
    try {
      const W = image.naturalWidth || image.width;
      const H = image.naturalHeight || image.height;
      if (!W || !H) return null;
      const c = document.createElement('canvas');
      const N = 48;                       // 축소해서 읽는다 — 정밀도가 필요 없다
      c.width = N; c.height = N;
      const g = c.getContext('2d');
      // 3장 촬영이면 텍스처가 아틀라스다. 가운데를 그냥 읽으면 앞면/옆면 **경계**를
      // 읽게 되어 피부톤이 엉뚱해진다. 앞면 칸만 잘라서 본다.
      if (frontRegion) {
        g.drawImage(image, frontRegion.x * W, frontRegion.y * H,
                    frontRegion.w * W, frontRegion.h * H, 0, 0, N, N);
      } else {
        g.drawImage(image, 0, 0, N, N);
      }
      const d = g.getImageData(0, 0, N, N).data;
      let r = 0, gg = 0, b = 0, n = 0;
      for (let y = Math.floor(N * 0.30); y < N * 0.72; y++) {
        for (let x = Math.floor(N * 0.28); x < N * 0.72; x++) {
          const i = (y * N + x) * 4;
          const lum = (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
          if (lum < 28 || lum > 242) continue;      // 그림자·하이라이트 제외
          r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++;
        }
      }
      if (n < 40) return null;
      return ((Math.round(r / n) << 16) | (Math.round(gg / n) << 8) | Math.round(b / n));
    } catch (e) {
      return null;    // 다른 출처 이미지 등으로 캔버스가 오염되면 읽을 수 없다
    }
  }

  /**
   * 3D 얼굴 생성.
   * @param {Array}  opts.landmarks  Face Mesh 468 랜드마크 (x,y,z 정규화 좌표)
   * @param {Canvas|Image} opts.image  같은 프레임의 얼굴 사진 (텍스처)
   * @param {number} opts.color       파이터 색 (림 라이트/코피 톤에 사용)
   * @param {number} opts.width       머리 폭 (휴머노이드 스케일, 기본 2.6)
   */
  window.createFace3D = function (opts) {
    const tris = getTriangles();
    if (!tris || !opts || !opts.landmarks || opts.landmarks.length < 400) return null;

    const lm = opts.landmarks;
    const width = opts.width || 2.6;
    const color = opts.color || 0xffffff;

    const basePos = normalizeLandmarks(lm, width, opts.aspect || 1);
    const index = fixWinding(basePos, tris.slice());

    // 눈·입은 테셀레이션에 **구멍**으로 남아 있다. 여기서 찾아 두고 아래에서 메운다.
    const holeLoops = findHoles(index);

    // UV = 랜드마크의 2D 화면 좌표. 사진과 랜드마크가 **같은 프레임**에서 나왔으므로
    // 이 매핑이 곧 정확한 텍스처 정합이 된다 (얼굴 정렬 과정이 따로 필요 없다).
    //
    // 텍스처는 얼굴만 잘라 쓴다. 전체 프레임을 그대로 쓰면 얼굴이 텍스처의 1/3 도 안 되어
    // 해상도를 그만큼 버리게 된다. crop 이 주어지면 그 사각형 기준으로 UV 를 다시 잡는다.
    const crop = opts.crop;   // { x0, y0, w, h } — 원본 이미지 픽셀 단위
    const IW = opts.imageW || 1, IH = opts.imageH || 1;
    // UV 는 **텍스처를 딴 바로 그 프레임의 랜드마크**로 계산해야 한다.
    // 형태(geometry)는 여러 프레임을 평균내 안정화하지만, 그 평균 좌표로 UV 를 잡으면
    // 텍스처와 미세하게 어긋나 눈·입이 밀린 "남의 얼굴"처럼 보인다.
    const uvLm = (opts.uvLandmarks && opts.uvLandmarks.length === lm.length) ? opts.uvLandmarks : lm;
    const uv = new Float32Array(lm.length * 2);
    for (let i = 0; i < lm.length; i++) {
      let u = uvLm[i].x, v = uvLm[i].y;
      if (crop) {
        u = (uvLm[i].x * IW - crop.x0) / crop.w;
        v = (uvLm[i].y * IH - crop.y0) / crop.h;
      }
      uv[i * 2] = u;
      uv[i * 2 + 1] = 1 - v;
    }
    // 3장 촬영이면 텍스처가 아틀라스다 — 앞면 UV 를 front 칸 안으로 접어 넣는다.
    // (1장 촬영일 때는 atlas 가 없고 텍스처 전체가 앞면이라 그대로 둔다.)
    if (opts.atlas && opts.atlas.front) {
      const R = opts.atlas.front;
      for (let i = 0; i < lm.length; i++) {
        uv[i * 2] = R.x + uv[i * 2] * R.w;
        uv[i * 2 + 1] = 1 - (R.y + (1 - uv[i * 2 + 1]) * R.h);
      }
    }

    // 얼굴의 실제 크기·깊이. 머리에 붙일 때 "구 안에 파묻히지 않게" 배치하려면
    // 얼굴이 z로 어디서 어디까지 뻗는지 호출부가 알아야 한다.
    const bounds = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity,
                     zMin: Infinity, zMax: -Infinity };
    for (let i = 0; i < lm.length; i++) {
      const x = basePos[i * 3], y = basePos[i * 3 + 1], z = basePos[i * 3 + 2];
      if (x < bounds.xMin) bounds.xMin = x;
      if (x > bounds.xMax) bounds.xMax = x;
      if (y < bounds.yMin) bounds.yMin = y;
      if (y > bounds.yMax) bounds.yMax = y;
      if (z < bounds.zMin) bounds.zMin = z;
      if (z > bounds.zMax) bounds.zMax = z;
    }

    // ── 뒤통수를 만들어 닫힌 머리로 ──────────────────────────────────
    //
    // Face Mesh 는 얼굴 **앞면만** 덮는다. 그대로 쓰면 어느 각도에서 봐도 "판자를 붙인 것"이고,
    // 옆·뒤에서는 뒤통수가 없어 속이 비어 보인다. 구를 뒤에 놓아 가려도 색과 형태가 끊긴다.
    //
    // 그래서 얼굴 테두리(FACE_OVAL 36점)에서 출발해 **뒤로 쓸어 넘겨 두개골을 만든다.**
    // 테두리를 적도로 보고 뒤쪽 극점까지 위도를 나눠 링을 쌓으면 닫힌 반구가 된다.
    // 테두리에서 시작하므로 이음매가 정확히 맞고, 사람마다 다른 얼굴 윤곽을 그대로 따라간다.
    // **뒤통수를 만들지 않는다.**
    //
    // 얼굴 테두리(FACE_OVAL)에서 두개골을 쓸어 만들어 봤지만, 실제 얼굴에서는
    // 뒤로 뾰족한 혹처럼 튀어나왔다. 테두리 모양·깊이가 사람마다 크게 달라
    // 상수 몇 개로 맞출 수 있는 문제가 아니다. 사진에 없는 것을 지어내는 일이라
    // 어느 각도에서는 반드시 티가 난다.
    //
    // 대신 humanoid 의 **구형 두개골**을 그대로 두고 얼굴만 앞면에 얹는다.
    // 머리 모양이 평범해서 어색하지 않고, 얼굴은 사진 그대로 나온다.
    //
    // 생성 코드는 아래에 그대로 남겨 둔다 — 되살리려면 이 상수만 true 로 바꾸면 된다.
    const BUILD_CRANIUM = false;
    const oval = BUILD_CRANIUM ? getFaceOval() : null;
    const RINGS = 9;                       // 적도 → 극점 사이 링 개수
    const BACK_DEPTH = 1.76;               // 뒤통수 깊이 (테두리 반지름 대비)
    // 정수리 높이. FACE_OVAL 의 맨 위는 이마 헤어라인이라 그 위 두개골이 통째로 없다.
    // 극점을 위로도 보내야 머리가 납작해지지 않는다.
    const VAULT_UP = 0.92;
    // 옆사진에서 머리(머리카락 포함)가 얼굴 랜드마크 범위보다 얼마나 더 넓은가
    const HEAD_EXPAND = 1.30;
    // 옆사진이 실제로 담고 있는 범위. 이보다 뒤는 어느 사진에도 안 찍혀 있으므로
    // 투영을 시도하지 않고 머리카락 색으로 채운다.
    const SIDE_MAX_T = 0.55;
    const craniumPos = [];                 // 추가 정점 (x,y,z ...)
    const craniumUV = [];
    const craniumCol = [];
    let craniumStart = lm.length;

    if (oval) {
      const n = oval.length;
      // 테두리의 중심과 평균 z — 이 지점을 두개골의 중심으로 삼는다
      let cx = 0, cy = 0, cz = 0;
      for (const idx of oval) {
        cx += basePos[idx * 3]; cy += basePos[idx * 3 + 1]; cz += basePos[idx * 3 + 2];
      }
      cx /= n; cy /= n; cz /= n;

      // 테두리 평균 반지름 — 두개골 크기의 기준
      let radMean = 0;
      for (const idx of oval) {
        radMean += Math.hypot(basePos[idx * 3] - cx, basePos[idx * 3 + 1] - cy);
      }
      radMean /= n;

      // **머리카락 UV.** FACE_OVAL 의 맨 위는 이마 헤어라인이다. 그 위쪽 픽셀이 머리카락이고
      // 촬영 크롭에 얼굴 높이의 18% 여백이 들어 있으므로 텍스처 안에 실제로 존재한다.
      // 뒤통수 색을 여기로 수렴시킨다 — 얼굴 안쪽(피부)으로 수렴시키면 뒤통수가
      // "민머리에 살색"이 되어 사람으로 안 보인다.
      let topIdx = oval[0], topY = Infinity;
      for (const idx of oval) {
        if (basePos[idx * 3 + 1] > topY) continue;      // y 는 위가 +
        // Three 좌표에서 위쪽 = y 큰 쪽
      }
      topY = -Infinity;
      for (const idx of oval) {
        if (basePos[idx * 3 + 1] > topY) { topY = basePos[idx * 3 + 1]; topIdx = idx; }
      }
      // 텍스처에서 이마 위쪽으로 조금 더 올라간 지점 (uv.y 는 뒤집혀 있어 위가 큰 값)
      let midU = 0, midV = 0;
      for (let i = 0; i < lm.length; i++) { midU += uv[i * 2]; midV += uv[i * 2 + 1]; }
      midU /= lm.length; midV /= lm.length;
      const hairU = uv[topIdx * 2];
      const hairV = Math.min(0.99, uv[topIdx * 2 + 1] + (uv[topIdx * 2 + 1] - midV) * 0.55);

      // ── 옆모습 3장 촬영이 있으면 두개골 UV 를 거기서 가져온다 ──────────
      //
      // 정면 1장뿐이면 뒤통수에 붙일 픽셀이 아예 없어 이마 위 머리카락을 늘여 붙이게 되고,
      // 그래서 모자를 쓴 것처럼 보인다. 옆모습에는 관자놀이·귀·귀 뒤 머리카락이 실제로 찍혀 있다.
      //
      // 두개골 정점은 사진에 랜드마크가 없다. 대신 **같은 얼굴의 468개 대응점**으로
      // 3D -> 옆사진 투영을 역산해 두면, 그 자리의 픽셀을 계산해 찍을 수 있다.
      const atlas = opts.atlas || null;
      const sv = opts.sideViews || null;
      const sideProj = { neg: null, pos: null };
      if (atlas && sv) {
        for (const key of ['neg', 'pos']) {
          const view = sv[key];
          const reg = atlas[key === 'neg' ? 'sideNeg' : 'sidePos'];
          if (!view || !view.lm || !reg) continue;
          const fit = fitProjection(basePos, view.lm, Math.min(lm.length, view.lm.length));
          if (!fit) continue;
          const vIW = view.imageW || 1, vIH = view.imageH || 1, vc = view.crop;

          // **그 사진에서 머리가 차지하는 타원**을 재 둔다.
          //
          // 아핀 투영은 얼굴 랜드마크로 맞춘 것이라 두개골(랜드마크가 없는 자리)로
          // 외삽하면 얼마든지 멀리 날아간다. 실제로 천장과 벽이 뒤통수에 발렸다.
          // 랜드마크 범위를 머리카락만큼 넓힌 타원 밖은 **사진에 머리가 없는 자리**이므로
          // 쓰지 않는다.
          let mu = 0, mv = 0;
          const M = Math.min(lm.length, view.lm.length);
          for (let i = 0; i < M; i++) { mu += view.lm[i].x; mv += view.lm[i].y; }
          mu /= M; mv /= M;
          let ru = 0, rv = 0;
          for (let i = 0; i < M; i++) {
            ru = Math.max(ru, Math.abs(view.lm[i].x - mu));
            rv = Math.max(rv, Math.abs(view.lm[i].y - mv));
          }
          sideProj[key] = { fit, reg, vIW, vIH, vc, mu, mv, ru, rv };
        }
      }

      /** 3D 점을 옆사진 아틀라스 UV 로. 범위를 벗어나면 null (그 view 로는 안 보이는 자리) */
      function sideUV(key, x, y, z) {
        const P = sideProj[key];
        if (!P) return null;
        const { a, b } = P.fit;
        let u = a[0] * x + a[1] * y + a[2] * z + a[3];
        let v = b[0] * x + b[1] * y + b[2] * z + b[3];

        // **머리 실루엣 안인가.** 밖이면 그 자리에 찍힌 것은 천장·벽이다.
        // 머리카락이 얼굴 랜드마크보다 바깥으로 나오므로 HEAD_EXPAND 만큼 넓혀 본다.
        const du = (u - P.mu) / (P.ru * HEAD_EXPAND);
        const dv = (v - P.mv) / (P.rv * HEAD_EXPAND);
        if (du * du + dv * dv > 1) return null;

        if (P.vc) {          // 사진 전체 -> 잘라낸 사각형 기준
          u = (u * P.vIW - P.vc.x0) / P.vc.w;
          v = (v * P.vIH - P.vc.y0) / P.vc.h;
        }
        if (u < 0.02 || u > 0.98 || v < 0.02 || v > 0.98) return null;
        // 아틀라스 안 그 view 의 칸으로 (uv.y 는 뒤집혀 있다)
        return [P.reg.x + u * P.reg.w, 1 - (P.reg.y + v * P.reg.h)];
      }

      // 링별 정점 생성. t=0 은 테두리(이미 존재), t=1 은 극점.
      //
      // 극점을 **뒤쪽만이 아니라 위쪽으로도** 보낸다. FACE_OVAL 의 맨 위가 헤어라인이라
      // 뒤로만 쓸어 넘기면 정수리가 통째로 없는 납작한 머리가 된다(실측 높이/폭 0.97,
      // 사람은 약 1.25). 극점을 위·뒤로 두면 이마 위로 두개골이 솟는다.
      // t 가 1 에 닿으면 shrink = cos(pi/2) = 0 이라 링 전체가 극점으로 뭉쳐
      // 면적 0 인 삼각형이 한 밴드(2 x n) 통째로 생긴다. 극점은 따로 두고
      // 링은 그 앞까지만 만든다.
      for (let r = 1; r <= RINGS; r++) {
        const t = r / (RINGS + 1);
        const shrink = Math.cos(t * Math.PI / 2);          // 1 → 0
        const sweep = Math.sin(t * Math.PI / 2);           // 0 → 1
        // 링 중심이 위·뒤로 이동한다.
        //
        // 뒤로는 단조롭게 가지만 **위로는 중간에서 가장 높고 그 뒤로 살짝 내려온다.**
        // 단조 상승으로 두면 머리의 최고점이 뒤통수 극점이 되어 뒤로 뾰족한 원뿔이
        // 된다. 사람은 정수리가 귀 위쯤에서 가장 높고 뒤통수는 그보다 낮다.
        const ringY = cy + VAULT_UP * radMean * Math.sin(t * Math.PI * 0.80);
        const ringZ = cz - BACK_DEPTH * radMean * sweep;
        for (let i = 0; i < n; i++) {
          const idx = oval[i];
          const ox = basePos[idx * 3] - cx;
          const oy = basePos[idx * 3 + 1] - cy;
          craniumPos.push(cx + ox * shrink, ringY + oy * shrink, ringZ);
          // UV — 옆모습이 있으면 그쪽 실제 픽셀을 쓰고, 없으면 이마 위 머리카락으로 수렴시킨다.
          // 좌우는 정점의 x 부호로 가른다 (얼굴 중심 기준).
          const px = cx + ox * shrink, py = ringY + oy * shrink, pz = ringZ;
          const key = (px < cx) ? 'neg' : 'pos';
          // 뒤통수 깊숙한 곳은 어느 사진에도 안 찍혀 있다 — 시도조차 하지 않는다
          let su = (t <= SIDE_MAX_T) ? sideUV(key, px, py, pz) : null;
          if (!su && t <= SIDE_MAX_T) su = sideUV(key === 'neg' ? 'pos' : 'neg', px, py, pz);
          if (su) {
            // 이음매(t 가 작을 때)에서는 앞면 UV 와 섞어 경계가 튀지 않게 한다
            const bw = Math.min(1, t * 2.2);
            craniumUV.push(uv[idx * 2] * (1 - bw) + su[0] * bw,
                           uv[idx * 2 + 1] * (1 - bw) + su[1] * bw);
          } else {
            // 옆사진이 못 미치는 자리 — 머리카락 색으로 수렴시킨다.
            // 옆사진 범위(SIDE_MAX_T)를 넘어가면 **완전히** 머리카락색으로 간다.
            // 조금이라도 테두리 UV 가 섞여 있으면 얼굴 가장자리(피부색)가 뒤통수까지
            // 끌려와 "살색 뒤통수"가 된다.
            const w = (t > SIDE_MAX_T) ? 1 : Math.min(1, t / SIDE_MAX_T);
            craniumUV.push(uv[idx * 2] * (1 - w) + hairU * w,
                           uv[idx * 2 + 1] * (1 - w) + hairV * w);
          }
          // 뒤로 갈수록 살짝 어둡게 — 빛이 덜 드는 자리라 이게 있어야 구형으로 읽힌다
          const sh = 1 - t * 0.26;
          craniumCol.push(sh, sh, sh);
        }
      }
      // 마지막 극점 하나 — 위·뒤로
      // 극점 — 정수리 곡선의 끝(중간보다 낮다)이자 가장 뒤
      craniumPos.push(cx,
                      cy + VAULT_UP * radMean * Math.sin(Math.PI * 0.80),
                      cz - BACK_DEPTH * radMean);
      // 극점(뒤통수 한가운데)은 어느 사진에도 안 찍혀 있다 — 머리카락 색으로 채운다
      craniumUV.push(hairU, hairV);
      craniumCol.push(0.70, 0.70, 0.70);
    }

    const geo = new THREE.BufferGeometry();
    const livePos = new Float32Array(basePos);          // 변형이 적용된 실제 정점
    // 얼굴 + 두개골을 하나의 버퍼로 합친다.
    // 구멍마다 중심 정점을 하나씩 추가해 부채꼴로 메운다.
    const holeStart = lm.length + craniumPos.length / 3;
    const totalV = holeStart + holeLoops.length;
    const allPos = new Float32Array(totalV * 3);
    const allUV = new Float32Array(totalV * 2);
    allPos.set(livePos, 0);
    allUV.set(uv, 0);
    for (let i = 0; i < craniumPos.length; i++) allPos[lm.length * 3 + i] = craniumPos[i];
    for (let i = 0; i < craniumUV.length; i++) allUV[lm.length * 2 + i] = craniumUV[i];

    // 두개골 삼각형 — 링과 링 사이를 사각형으로 잇고 둘로 쪼갠다
    const craniumTriStart = index.length;
    if (oval) {
      const n = oval.length;
      const ringIdx = (r, i) => craniumStart + (r - 1) * n + (i % n);   // r >= 1
      const rimIdx = (i) => oval[i % n];
      for (let i = 0; i < n; i++) {
        // 테두리 ↔ 첫 링
        index.push(rimIdx(i), ringIdx(1, i), rimIdx(i + 1));
        index.push(rimIdx(i + 1), ringIdx(1, i), ringIdx(1, i + 1));
        // 링 사이
        for (let r = 1; r < RINGS; r++) {
          index.push(ringIdx(r, i), ringIdx(r + 1, i), ringIdx(r, i + 1));
          index.push(ringIdx(r, i + 1), ringIdx(r + 1, i), ringIdx(r + 1, i + 1));
        }
        // 마지막 링 ↔ 극점
        const pole = craniumStart + RINGS * n;
        index.push(ringIdx(RINGS, i), pole, ringIdx(RINGS, i + 1));
      }
    }

    // ── 눈·입 구멍 메우기 ────────────────────────────────────────────
    //
    // 각 루프의 중심에 정점을 하나 놓고 부채꼴로 잇는다. 중심은 **살짝 안쪽**으로
    // 밀어 넣는다 — 평면으로 메우면 눈이 스티커처럼 붙지만, 안으로 들어가면
    // 눈두덩 그늘이 생겨 안구가 들어앉은 것처럼 읽힌다. 입도 같은 이유로 오목하게.
    //
    // UV 는 루프 UV 의 평균이다. 그 자리가 사진에서 정확히 눈동자·입 안쪽이므로
    // 텍스처가 그대로 얹힌다. (z 는 카메라 쪽이 +. 안쪽 = 더 작은 z)
    const HOLE_SINK = 0.30;         // 중심을 얼마나 밀어 넣을지 (구멍 반지름 대비)
    const holeInfo = [];            // update() 가 변형을 따라가게 하려고 보관
    holeLoops.forEach((loop, h) => {
      const vi = holeStart + h;
      let cx = 0, cy = 0, cz = 0, cu = 0, cv = 0;
      for (const idx of loop) {
        cx += allPos[idx * 3]; cy += allPos[idx * 3 + 1]; cz += allPos[idx * 3 + 2];
        cu += allUV[idx * 2];  cv += allUV[idx * 2 + 1];
      }
      const n = loop.length;
      cx /= n; cy /= n; cz /= n; cu /= n; cv /= n;
      // 구멍 반지름 = 중심에서 테두리까지 평균 거리
      let rad = 0;
      for (const idx of loop) {
        rad += Math.hypot(allPos[idx * 3] - cx, allPos[idx * 3 + 1] - cy);
      }
      rad /= n;
      const sink = rad * HOLE_SINK;
      allPos[vi * 3] = cx; allPos[vi * 3 + 1] = cy; allPos[vi * 3 + 2] = cz - sink;
      allUV[vi * 2] = cu;  allUV[vi * 2 + 1] = cv;
      holeInfo.push({ loop, sink, vi });

      // 부채꼴. 감기 방향은 얼굴 나머지와 맞춰야 앞면이 뒤집히지 않는다.
      const fan = [];
      for (let i = 0; i < n; i++) fan.push(vi, loop[i], loop[(i + 1) % n]);
      fixWinding(allPos, fan);
      for (const t of fan) index.push(t);
    });

    // ── 두개골 감기 방향 ──────────────────────────────────────────────
    //
    // fixWinding 은 **법선의 z 성분**으로 판단한다. 얼굴 앞면에는 맞지만 뒤통수는
    // 법선이 -z 라 그 기준을 그대로 쓰면 통째로 뒤집힌다. 실제로 그랬다 —
    // 뒤쪽 삼각형 468개 중 450개가 안쪽을 향했고, FrontSide 라 뒤통수가
    // 아예 안 그려졌다("뒤가 뚫려 보인다"의 정체).
    //
    // 닫힌 볼록면이므로 **머리 중심에서 바깥으로** 향하는지로 판단한다.
    if (oval && craniumTriStart < index.length) {
      let hx = 0, hy = 0, hz = 0;
      const nv = holeStart;                       // 얼굴 + 두개골 정점
      for (let i = 0; i < nv; i++) { hx += allPos[i*3]; hy += allPos[i*3+1]; hz += allPos[i*3+2]; }
      hx /= nv; hy /= nv; hz /= nv;
      for (let t = craniumTriStart; t + 2 < index.length; t += 3) {
        const i = index[t], j = index[t+1], k = index[t+2];
        const ax = allPos[i*3], ay = allPos[i*3+1], az = allPos[i*3+2];
        const bx = allPos[j*3], by = allPos[j*3+1], bz = allPos[j*3+2];
        const kx = allPos[k*3], ky = allPos[k*3+1], kz = allPos[k*3+2];
        const ux = bx-ax, uy = by-ay, uz = bz-az;
        const vx = kx-ax, vy = ky-ay, vz = kz-az;
        const nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx;
        const gx = (ax+bx+kx)/3 - hx, gy = (ay+by+ky)/3 - hy, gz = (az+bz+kz)/3 - hz;
        if (nx*gx + ny*gy + nz*gz < 0) { index[t+1] = k; index[t+2] = j; }
      }
    }

    geo.setAttribute('position', new THREE.BufferAttribute(allPos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(allUV, 2));
    // 멍/홍조를 정점 색으로 누적한다 — 텍스처를 다시 그리지 않아도 손상이 보인다
    const vcol = new Float32Array(totalV * 3).fill(1);
    for (let i = 0; i < craniumCol.length; i++) vcol[lm.length * 3 + i] = craniumCol[i];
    geo.setAttribute('color', new THREE.BufferAttribute(vcol, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();

    let tex = null;
    let skinTone = null;
    if (opts.image) {
      // **머리 바깥을 머리카락색으로 덮은 사본**을 텍스처로 쓴다.
      // 그러지 않으면 두개골 UV 가 천장·벽을 찍는다(실제로 그랬다).
      const regions = opts.atlas || { front: { x: 0, y: 0, w: 1, h: 1 } };
      // **좌표계를 맞춘다.**
      //
      // 랜드마크는 카메라 **전체 프레임** 기준 0~1 인데, 텍스처는 얼굴만 잘라낸 사진이다.
      // 그대로 넘기면 머리 타원이 엉뚱한 자리에 그려져 **얼굴 대부분을 머리카락색으로
      // 덮어버린다** — 실제로 눈·코만 남고 나머지가 까맣게 나왔다.
      // UV 를 계산할 때와 똑같이 crop 을 통과시켜 잘라낸 사진 기준으로 바꿔 준다.
      const toCropSpace = (arr, cp, iw, ih) => {
        if (!arr) return null;
        if (!cp) return arr;
        return arr.map(q => ({
          x: (q.x * iw - cp.x0) / cp.w,
          y: (q.y * ih - cp.y0) / cp.h,
        }));
      };
      const sn = opts.sideViews && opts.sideViews.neg;
      const sp = opts.sideViews && opts.sideViews.pos;
      const lmByKey = {
        front: toCropSpace(opts.uvLandmarks || lm, crop, IW, IH),
        sideNeg: sn ? toCropSpace(sn.lm, sn.crop, sn.imageW || 1, sn.imageH || 1) : null,
        sidePos: sp ? toCropSpace(sp.lm, sp.crop, sp.imageW || 1, sp.imageH || 1) : null,
      };
      const painted = paintOutsideHead(opts.image, regions, lmByKey, HEAD_EXPAND);
      const src = painted ? painted.canvas : opts.image;
      tex = new THREE.CanvasTexture(src);
      tex.flipY = true;
      tex.needsUpdate = true;
      skinTone = sampleSkinTone(src, opts.atlas && opts.atlas.front);
    }

    const mat = new THREE.MeshStandardMaterial({
      map: tex,
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.72,
      metalness: 0.02,
      // 두개골까지 닫힌 메쉬이므로 앞면만 그리면 된다.
      // DoubleSide 로 두면 안쪽 면이 비쳐 얼굴 안이 들여다보인다.
      side: THREE.FrontSide,
      emissive: color,
      emissiveIntensity: 0.06,     // 파이터 색을 아주 옅게 얹어 아바타와 톤을 맞춘다
    });

    const mesh = new THREE.Mesh(geo, mat);

    // ── 코피 ────────────────────────────────────────────────────────────
    if (!bloodTex) bloodTex = makeBloodTexture();
    const bloodMat = new THREE.MeshBasicMaterial({
      map: bloodTex, transparent: true, opacity: 0, depthWrite: false,
      side: THREE.DoubleSide,
    });
    const blood = new THREE.Mesh(new THREE.PlaneGeometry(width * 0.16, width * 0.42), bloodMat);
    // 콧구멍 아래에 붙인다. 위치는 랜드마크에서 직접 가져오므로 얼굴 형태를 따라간다.
    const nb = IDX.NOSE_BOTTOM;
    blood.position.set(basePos[nb * 3] * 0.6,
                       basePos[nb * 3 + 1] - width * 0.16,
                       basePos[nb * 3 + 2] + 0.02);
    mesh.add(blood);

    // ── 상태 ────────────────────────────────────────────────────────────
    const S = {
      hp: 100,
      impacts: [],        // { idx, dir, amp, t, life } — 눌린 자국
      swellTarget: 0,     // 누적 부기 (HP가 낮을수록 크다)
      swell: 0,
      bloodAmt: 0,
      shake: 0,
      expr: 0,            // 0(멀쩡) → 1(탈진). HP에서 유도
      exprNow: 0,
    };

    // 표정용 오프셋을 미리 계산해 둔다 (매 프레임 인덱스 순회를 피한다)
    const exprOffsets = buildExpressionOffsets(basePos, width);

    /**
     * 지친 표정 오프셋. 사람이 지치면 나타나는 변화를 부위별로 넣는다.
     *  - 눈썹 안쪽이 올라가고 바깥이 처진다 (팔자 눈썹)
     *  - 윗눈꺼풀이 내려온다 (눈이 반쯤 감김)
     *  - 입이 벌어진다 (헐떡임)
     *  - 턱이 살짝 내려간다
     */
    function buildExpressionOffsets(pos, w) {
      const off = new Float32Array(pos.length);   // 정점별 (dx,dy,dz)
      const add = (i, dx, dy, dz) => {
        off[i * 3] += dx; off[i * 3 + 1] += dy; off[i * 3 + 2] += dz;
      };
      // 눈썹 — 안쪽(배열 앞)일수록 올리고 바깥쪽일수록 내린다
      const brow = (arr, sign) => arr.forEach((id, k) => {
        const t = k / Math.max(1, arr.length - 1);          // 0=안쪽, 1=바깥쪽
        add(id, sign * w * 0.005, w * (0.030 - 0.055 * t), 0);
      });
      brow(IDX.BROW_L, -1);
      brow(IDX.BROW_R, 1);
      // 윗눈꺼풀 내림 (눈이 감긴다)
      IDX.LID_UPPER_L.concat(IDX.LID_UPPER_R).forEach(id => add(id, 0, -w * 0.030, 0));
      IDX.LID_LOWER_L.concat(IDX.LID_LOWER_R).forEach(id => add(id, 0, w * 0.008, 0));
      // 입 벌어짐 — 아랫입술과 턱을 내린다
      IDX.LIP_LOWER.forEach(id => add(id, 0, -w * 0.055, 0));
      IDX.LIP_UPPER.forEach(id => add(id, 0, w * 0.010, 0));
      add(IDX.CHIN, 0, -w * 0.070, 0);
      // 입꼬리 처짐
      add(IDX.MOUTH_CORNER_L, -w * 0.012, -w * 0.030, 0);
      add(IDX.MOUTH_CORNER_R, w * 0.012, -w * 0.030, 0);
      return off;
    }

    /** 피격 지점을 고른다. 방향이 주어지면 그쪽 뺨, 아니면 코 주변. */
    function pickImpactIndex(side) {
      if (side === 'left') return IDX.CHEEK_L;
      if (side === 'right') return IDX.CHEEK_R;
      if (side === 'chin') return IDX.CHIN;
      return IDX.NOSE_TIP;
    }

    /**
     * 피격 — 그 지점이 안쪽으로 눌렸다가 탄성으로 돌아오고, 멍이 남는다.
     * @param {number} damage 데미지 (변형 세기)
     * @param {string} side   'left' | 'right' | 'chin' | 'center'
     */
    function hit(damage, side) {
      const idx = pickImpactIndex(side);
      const amp = Math.min(1.0, 0.35 + (damage || 5) / 14) * width * 0.13;
      S.impacts.push({ idx, amp, t: 0, life: 0.7 });
      if (S.impacts.length > 6) S.impacts.shift();
      S.shake = Math.min(1, S.shake + 0.5 + (damage || 5) / 20);

      // 멍 — 맞은 자리 주변 정점을 붉게/보랏빛으로 물들인다. 지워지지 않고 누적된다.
      const cx = basePos[idx * 3], cy = basePos[idx * 3 + 1], cz = basePos[idx * 3 + 2];
      const R = width * 0.30, R2 = R * R;
      const strength = Math.min(0.5, 0.12 + (damage || 5) / 60);
      for (let i = 0; i < lm.length; i++) {
        const dx = basePos[i * 3] - cx, dy = basePos[i * 3 + 1] - cy, dz = basePos[i * 3 + 2] - cz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > R2) continue;
        const f = (1 - Math.sqrt(d2) / R) * strength;
        vcol[i * 3]     = Math.max(0.35, vcol[i * 3] - f * 0.15);   // R 은 덜 깎아 붉게 남긴다
        vcol[i * 3 + 1] = Math.max(0.20, vcol[i * 3 + 1] - f);
        vcol[i * 3 + 2] = Math.max(0.28, vcol[i * 3 + 2] - f * 0.75);
      }
      geo.attributes.color.needsUpdate = true;
    }

    /** HP 반영 — 낮을수록 지친 표정, 부기, 코피. */
    function setHp(hp) {
      const v = Math.max(0, Math.min(100, hp));
      // 코피는 60 이하부터 시작해 0 에서 최대. HP 바와 별개로 "몸 상태"가 얼굴에 드러난다.
      S.hp = v;
      S.expr = 1 - v / 100;
      S.swellTarget = (1 - v / 100) * width * 0.045;
      S.bloodAmt = v >= 60 ? 0 : Math.min(1, (60 - v) / 45);
    }

    const _tmpNormal = new THREE.Vector3();
    function update(dt) {
      const n = lm.length;      // 앞쪽 얼굴 정점만 변형한다 — 두개골은 고정
      const pos = geo.attributes.position.array;

      // 표정 — 목표치로 부드럽게 수렴 (급변하면 표정이 튄다)
      S.exprNow += (S.expr - S.exprNow) * Math.min(1, dt * 3.0);
      S.swell += (S.swellTarget - S.swell) * Math.min(1, dt * 2.0);
      S.shake = Math.max(0, S.shake - dt / 0.45);

      // 피격 자국 감쇠 — 눌렸다가 스프링처럼 되돌아온다 (약간 튕기며)
      for (let k = S.impacts.length - 1; k >= 0; k--) {
        const im = S.impacts[k];
        im.t += dt;
        if (im.t >= im.life) S.impacts.splice(k, 1);
      }

      // 헐떡임 — 지칠수록 크고 빠르게 (부기와 함께 얼굴이 미세하게 움직인다)
      const t = performance.now() * 0.001;
      const pant = S.exprNow * Math.sin(t * (3.2 + S.exprNow * 3.0)) * width * 0.010;

      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        let x = basePos[i3], y = basePos[i3 + 1], z = basePos[i3 + 2];

        // 1) 표정
        x += exprOffsets[i3] * S.exprNow;
        y += exprOffsets[i3 + 1] * S.exprNow + pant * 0.35;
        z += exprOffsets[i3 + 2] * S.exprNow;

        // 2) 부기 — 법선 방향으로 살짝 부풀린다 (얼굴이 붓는다)
        if (S.swell > 1e-4) {
          const nx = geo.attributes.normal.array[i3];
          const ny = geo.attributes.normal.array[i3 + 1];
          const nz = geo.attributes.normal.array[i3 + 2];
          x += nx * S.swell; y += ny * S.swell; z += nz * S.swell;
        }

        // 3) 피격 자국 — 가우시안 감쇠로 눌린 자국. 스프링 되돌림 포함.
        for (let k = 0; k < S.impacts.length; k++) {
          const im = S.impacts[k];
          const ci = im.idx * 3;
          const dx = basePos[i3] - basePos[ci];
          const dy = basePos[i3 + 1] - basePos[ci + 1];
          const dz = basePos[i3 + 2] - basePos[ci + 2];
          const R = width * 0.34;
          const d2 = (dx * dx + dy * dy + dz * dz) / (R * R);
          if (d2 > 4) continue;
          const falloff = Math.exp(-d2 * 1.6);
          // 0 → 최대로 훅 들어갔다가 감쇠 진동하며 복귀
          const u = im.t / im.life;
          const env = Math.exp(-u * 4.2) * Math.cos(u * Math.PI * 2.4);
          z -= im.amp * falloff * env;          // 안쪽(-z)으로 눌림
          y -= im.amp * falloff * env * 0.25;
        }

        pos[i3] = x; pos[i3 + 1] = y; pos[i3 + 2] = z;
      }

      // 눈·입 구멍의 중심 정점은 테두리를 따라간다. 고정해 두면 얼굴이 눌릴 때
      // 눈만 제자리에 남아 구멍 밖으로 삐져나온다.
      for (const h of holeInfo) {
        let cx = 0, cy = 0, cz = 0;
        for (const idx of h.loop) {
          cx += pos[idx * 3]; cy += pos[idx * 3 + 1]; cz += pos[idx * 3 + 2];
        }
        const k = h.loop.length;
        const v3 = h.vi * 3;
        pos[v3] = cx / k; pos[v3 + 1] = cy / k; pos[v3 + 2] = cz / k - h.sink;
      }

      geo.attributes.position.needsUpdate = true;
      // 법선은 매 프레임 다시 계산하면 비싸다. 변형이 있을 때만.
      if (S.impacts.length || S.swell > 1e-4 || S.exprNow > 1e-3) geo.computeVertexNormals();

      // 피격 직후 머리가 흔들린다
      mesh.rotation.z = Math.sin(t * 42) * S.shake * 0.09;
      mesh.rotation.x = Math.sin(t * 33) * S.shake * 0.05;

      // 코피 — 서서히 진해지고 아래로 흘러내린다
      bloodMat.opacity += (S.bloodAmt * 0.92 - bloodMat.opacity) * Math.min(1, dt * 1.5);
      blood.visible = bloodMat.opacity > 0.02;
      if (blood.visible) {
        blood.scale.y = 0.55 + S.bloodAmt * 0.95;
        blood.position.y = basePos[IDX.NOSE_BOTTOM * 3 + 1] - width * (0.10 + 0.10 * blood.scale.y);
      }
    }

    function dispose() {
      geo.dispose();
      mat.dispose();
      if (tex) tex.dispose();
      bloodMat.dispose();
      blood.geometry.dispose();
    }

    // **얼굴만의 바운딩** — 아바타에 붙일 때 이걸 기준으로 정렬한다.
    // 두개골까지 포함한 바운딩으로 중심을 맞추면, 정수리를 세운 만큼 중심이 위로
    // 올라가 얼굴이 아래로 처진다(턱이 가슴에 파묻힌다). 보는 사람이 위치를
    // 판단하는 기준은 **얼굴**이므로 얼굴을 머리 자리에 놓고 두개골은 뒤로 뻗게 둔다.
    const faceBounds = { xMin: bounds.xMin, xMax: bounds.xMax,
                         yMin: bounds.yMin, yMax: bounds.yMax,
                         zMin: bounds.zMin, zMax: bounds.zMax };

    // 두개골까지 포함한 실제 바운딩 (전체 크기 판단용)
    for (let i = 0; i < totalV; i++) {
      const x = allPos[i * 3], y = allPos[i * 3 + 1], z = allPos[i * 3 + 2];
      if (x < bounds.xMin) bounds.xMin = x;
      if (x > bounds.xMax) bounds.xMax = x;
      if (y < bounds.yMin) bounds.yMin = y;
      if (y > bounds.yMax) bounds.yMax = y;
      if (z < bounds.zMin) bounds.zMin = z;
      if (z > bounds.zMax) bounds.zMax = z;
    }

    return { mesh, hit, setHp, update, dispose, state: S, bounds, faceBounds, skinTone,
             // 뒤통수까지 닫힌 머리인가 — 호출부가 구 머리를 숨길지 판단한다
             isFullHead: !!oval,
             triangleCount: index.length / 3 };
  };

  // ── 직렬화 (네트워크 전송용) ───────────────────────────────────────────
  /**
   * 랜드마크 + 사진을 네트워크로 보낼 수 있는 형태로 만든다.
   * 좌표는 소수 4자리로 자른다 — 468×3 을 그대로 보내면 JSON 이 3배로 부푼다.
   */
  function flatten(landmarks) {
    const out = new Array(landmarks.length * 3);
    for (let i = 0; i < landmarks.length; i++) {
      out[i * 3]     = Math.round(landmarks[i].x * 10000) / 10000;
      out[i * 3 + 1] = Math.round(landmarks[i].y * 10000) / 10000;
      out[i * 3 + 2] = Math.round(landmarks[i].z * 10000) / 10000;
    }
    return out;
  }

  window.serializeFace = function (landmarks, canvas, meta, quality) {
    const m = meta || {};
    return {
      lm: flatten(landmarks),
      // 텍스처를 딴 프레임의 랜드마크 (UV 전용). 없으면 수신 측이 lm 을 그대로 쓴다.
      uvLm: m.uvLandmarks ? flatten(m.uvLandmarks) : null,
      tex: canvas.toDataURL('image/jpeg', quality || 0.72),
      aspect: m.aspect || 1,
      crop: m.crop || null,
      imageW: m.imageW || 1,
      imageH: m.imageH || 1,
      // 3장 촬영: 텍스처가 아틀라스이고, 옆모습 랜드마크로 두개골 UV 를 계산한다.
      // 1장 촬영이면 둘 다 없고 기존 경로가 그대로 돈다.
      atlas: m.atlas || null,
      sideViews: m.sideViews ? {
        neg: m.sideViews.neg ? { lm: flatten(m.sideViews.neg.lm), crop: m.sideViews.neg.crop,
                                 imageW: m.sideViews.neg.imageW, imageH: m.sideViews.neg.imageH } : null,
        pos: m.sideViews.pos ? { lm: flatten(m.sideViews.pos.lm), crop: m.sideViews.pos.crop,
                                 imageW: m.sideViews.pos.imageW, imageH: m.sideViews.pos.imageH } : null,
      } : null,
    };
  };

  /** 수신한 blob 으로 3D 얼굴을 만든다. 텍스처 로드가 끝나야 하므로 Promise. */
  window.createFace3DFromBlob = function (blob, color, width) {
    return new Promise((resolve) => {
      if (!blob || !blob.lm || !blob.tex) { resolve(null); return; }
      const unflatten = (arr) => {
        const out = [];
        for (let i = 0; i < arr.length; i += 3) out.push({ x: arr[i], y: arr[i + 1], z: arr[i + 2] });
        return out;
      };
      const landmarks = unflatten(blob.lm);
      const uvLandmarks = blob.uvLm ? unflatten(blob.uvLm) : null;
      const sv = blob.sideViews ? {
        neg: blob.sideViews.neg ? Object.assign({}, blob.sideViews.neg,
               { lm: unflatten(blob.sideViews.neg.lm) }) : null,
        pos: blob.sideViews.pos ? Object.assign({}, blob.sideViews.pos,
               { lm: unflatten(blob.sideViews.pos.lm) }) : null,
      } : null;
      const mk = (image) => window.createFace3D({
        landmarks, uvLandmarks, image, color, width,
        aspect: blob.aspect, crop: blob.crop, imageW: blob.imageW, imageH: blob.imageH,
        atlas: blob.atlas, sideViews: sv,
      });
      const img = new Image();
      img.onload = () => resolve(mk(img));
      img.onerror = () => resolve(mk(null));
      img.src = blob.tex;
    });
  };

  window.FACE3D_IDX = IDX;
  window.__faceTrianglesFromEdges = trianglesFromEdges;   // 하니스 검증용
})();
