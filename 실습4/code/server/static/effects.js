/**
 * effects.js — 타격 이펙트 (Three.js r128 전역 THREE 의존, ES module 아님)
 *
 * host 관제뷰(arena.html)와 1인칭 뷰(fighter_client.html)가 같은 연출을 써야 하므로
 * 한 곳에 모았다. 모든 이펙트는 카메라를 향해 빌보드되는 Sprite 기반이라
 * 두 뷰의 카메라 각도가 달라도 똑같이 보인다.
 *
 * 한 번의 타격은 9개 레이어가 겹쳐 "폭발"로 읽힌다:
 *   백색 코어 섬광 → 화염구 → 만화풍 스타버스트 → 충격파 링 2겹(속도 다름)
 *   → 연기 → 불꽃 파편 → HIT!/CRITICAL! 임팩트 텍스트 → 데미지 숫자
 *
 * 사용법:
 *   const fx = window.createHitEffects(scene);
 *   fx.spawnHit(new THREE.Vector3(x, 4.6, z), 0xff3366, 7, false);  // 위치, 색, 데미지, 가드여부
 *   fx.spawnKO(new THREE.Vector3(x, 5.2, z), 0xff3366);
 *   fx.update(dtSeconds);   // 매 프레임
 *   fx.clear();             // 라운드 리셋 등에서 즉시 정리
 */
(function () {
  if (typeof THREE === 'undefined') {
    console.error('[effects.js] THREE 로드 필요 (three.min.js 이후에 include)');
    return;
  }

  // ---------- 공용 텍스처 (한 번만 생성해 모든 이펙트가 공유) ----------
  let texGlow = null, texRing = null, texRingThin = null, texStar = null, texSmoke = null;

  function canvas(size) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    return c;
  }

  function makeGlowTexture() {
    const c = canvas(128), g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0.00, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.85)');
    grad.addColorStop(0.60, 'rgba(255,255,255,0.22)');
    grad.addColorStop(1.00, 'rgba(255,255,255,0)');
    g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  /** 가운데가 비고 테두리만 밝은 충격파 링. thin=true면 더 가늘고 날카롭다. */
  function makeRingTexture(thin) {
    const c = canvas(128), g = c.getContext('2d');
    const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    if (thin) {
      grad.addColorStop(0.00, 'rgba(255,255,255,0)');
      grad.addColorStop(0.78, 'rgba(255,255,255,0)');
      grad.addColorStop(0.88, 'rgba(255,255,255,1)');
      grad.addColorStop(0.95, 'rgba(255,255,255,0.25)');
      grad.addColorStop(1.00, 'rgba(255,255,255,0)');
    } else {
      grad.addColorStop(0.00, 'rgba(255,255,255,0)');
      grad.addColorStop(0.58, 'rgba(255,255,255,0)');
      grad.addColorStop(0.78, 'rgba(255,255,255,1)');
      grad.addColorStop(0.92, 'rgba(255,255,255,0.35)');
      grad.addColorStop(1.00, 'rgba(255,255,255,0)');
    }
    g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  /** 만화풍 스타버스트 — 길이가 들쭉날쭉한 스파이크가 사방으로 뻗는다. */
  function makeStarTexture() {
    const c = canvas(256), g = c.getContext('2d');
    const cx = 128, cy = 128, spikes = 14;
    g.translate(cx, cy);
    g.beginPath();
    for (let i = 0; i < spikes * 2; i++) {
      // 바깥/안쪽 반지름을 번갈아 찍어 뾰족한 별을 만든다. 살짝 불규칙해야 폭발처럼 보인다.
      const outer = i % 2 === 0;
      const r = outer ? 126 * (0.62 + 0.38 * Math.abs(Math.sin(i * 2.4))) : 34;
      const a = (i / (spikes * 2)) * Math.PI * 2;
      const x = Math.cos(a) * r, y = Math.sin(a) * r;
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.closePath();
    const grad = g.createRadialGradient(0, 0, 0, 0, 0, 126);
    grad.addColorStop(0.0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.5, 'rgba(255,255,255,0.9)');
    grad.addColorStop(1.0, 'rgba(255,255,255,0)');
    g.fillStyle = grad; g.fill();
    return new THREE.CanvasTexture(c);
  }

  /** 부드러운 연기 뭉치 — 폭발에 부피감을 준다. */
  function makeSmokeTexture() {
    const c = canvas(128), g = c.getContext('2d');
    for (let i = 0; i < 9; i++) {
      const a = (i / 9) * Math.PI * 2;
      const x = 64 + Math.cos(a) * 26, y = 64 + Math.sin(a) * 26;
      const r = 30 + (i % 3) * 8;
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0.0, 'rgba(255,255,255,0.55)');
      grad.addColorStop(1.0, 'rgba(255,255,255,0)');
      g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    }
    return new THREE.CanvasTexture(c);
  }

  // 텍스트 텍스처는 캔버스를 새로 그리는 비용이 있으므로 문자열 단위로 캐시한다.
  // 데미지 숫자는 종류가 몇 개 안 되어 캐시 적중률이 아주 높다.
  const textCache = new Map();

  function makeTextTexture(text, cssColor, fontPx, italic) {
    const key = `${text}|${cssColor}|${fontPx}|${italic ? 1 : 0}`;
    const hit = textCache.get(key);
    if (hit) return hit;

    const pad = 30;
    const font = `${italic ? 'italic ' : ''}900 ${fontPx}px "Segoe UI", Impact, sans-serif`;
    const probe = canvas(8).getContext('2d');
    probe.font = font;
    const w = Math.ceil(probe.measureText(text).width) + pad * 2;

    const c = document.createElement('canvas');
    c.width = THREE.MathUtils.ceilPowerOfTwo(w);
    c.height = THREE.MathUtils.ceilPowerOfTwo(fontPx + pad * 2);
    const g = c.getContext('2d');
    g.font = font;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.lineJoin = 'round';
    const cx = c.width / 2, cy = c.height / 2;

    // 검은 두꺼운 외곽선 → 밝은 폭발 위에서도 글자가 읽힌다
    g.strokeStyle = 'rgba(0,0,0,0.92)';
    g.lineWidth = Math.max(6, fontPx * 0.20);
    g.strokeText(text, cx, cy);
    // 위쪽이 밝고 아래가 어두운 금속 그라디언트
    const grad = g.createLinearGradient(0, cy - fontPx * 0.6, 0, cy + fontPx * 0.6);
    grad.addColorStop(0.0, '#ffffff');
    grad.addColorStop(0.45, cssColor);
    grad.addColorStop(1.0, '#ffffff');
    g.fillStyle = grad;
    g.fillText(text, cx, cy);

    const t = new THREE.CanvasTexture(c);
    t.aspect = c.width / c.height;
    textCache.set(key, t);
    return t;
  }

  function hexToCss(hex) {
    return '#' + ('000000' + (hex >>> 0).toString(16)).slice(-6);
  }

  /** 두 색을 섞는다 (화염구 중심을 흰색 쪽으로 당길 때 사용) */
  function mixHex(a, b, t) {
    const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
    const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
    return (Math.round(ar + (br - ar) * t) << 16)
         | (Math.round(ag + (bg - ag) * t) << 8)
         | Math.round(ab + (bb - ab) * t);
  }

  window.createHitEffects = function (scene) {
    if (!texGlow)     texGlow     = makeGlowTexture();
    if (!texRing)     texRing     = makeRingTexture(false);
    if (!texRingThin) texRingThin = makeRingTexture(true);
    if (!texStar)     texStar     = makeStarTexture();
    if (!texSmoke)    texSmoke    = makeSmokeTexture();

    const live = [];   // { obj, t, life, kind, ... }

    function addSprite(tex, color, pos, scale, opacity, blending, rot) {
      const mat = new THREE.SpriteMaterial({
        map: tex,
        color: color,
        transparent: true,
        opacity: opacity,
        blending: blending === undefined ? THREE.AdditiveBlending : blending,
        depthWrite: false,
        depthTest: false,     // 아바타에 가려지지 않게 — 타격은 항상 보여야 한다
        rotation: rot || 0,
      });
      const sp = new THREE.Sprite(mat);
      sp.position.copy(pos);
      sp.scale.set(scale, scale, 1);
      scene.add(sp);
      return sp;
    }

    /** 불꽃 파편. 사방으로 튀며 중력을 받는다. */
    function addSparks(pos, color, count, speed, size) {
      const geo = new THREE.BufferGeometry();
      const arr = new Float32Array(count * 3);
      const vel = [];
      for (let i = 0; i < count; i++) {
        arr[i * 3] = pos.x; arr[i * 3 + 1] = pos.y; arr[i * 3 + 2] = pos.z;
        // 구면 균등 분포로 튀는 방향을 뽑는다
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const sp = speed * (0.35 + Math.random() * 1.0);
        vel.push(new THREE.Vector3(
          Math.sin(ph) * Math.cos(th) * sp,
          Math.abs(Math.cos(ph)) * sp * 0.9 + speed * 0.3,
          Math.sin(ph) * Math.sin(th) * sp
        ));
      }
      geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      const mat = new THREE.PointsMaterial({
        color: color, size: size || 0.9, map: texGlow,
        transparent: true, opacity: 1, blending: THREE.AdditiveBlending,
        depthWrite: false, sizeAttenuation: true,
      });
      const pts = new THREE.Points(geo, mat);
      scene.add(pts);
      return { pts, vel };
    }

    function push(obj, life, kind, extra) {
      live.push(Object.assign({ obj, t: 0, life, kind, delay: 0 }, extra || {}));
    }

    /** 임팩트 텍스트 — 튀어나오며 흔들리고 떠오른다. */
    function addImpactText(pos, text, cssColor, fontPx, height, life, rise, tilt, delay) {
      const tex = makeTextTexture(text, cssColor, fontPx, true);
      const sp = addSprite(tex, 0xffffff, pos, 1, 0, THREE.NormalBlending, tilt);
      sp.scale.set(height * tex.aspect, height, 1);
      push(sp, life, 'text', { rise, tex, baseH: height, tilt, delay: delay || 0 });
      return sp;
    }

    /**
     * 타격 이펙트 — 9겹이 겹쳐 폭발로 읽힌다.
     * @param {THREE.Vector3} pos    타격 지점 (보통 피격자 가슴 높이)
     * @param {number} colorHex      공격자 색
     * @param {number} damage        데미지 (숫자로 띄운다)
     * @param {boolean} isGuard      가드로 막혔는지 — 막히면 푸른 톤에 작게
     */
    function spawnHit(pos, colorHex, damage, isGuard) {
      const color = isGuard ? 0x66ddff : colorHex;
      const hot   = mixHex(color, 0xffffff, 0.55);   // 화염구 안쪽은 흰색에 가깝게
      const mag   = isGuard ? 0.55 : 1.0;            // 가드면 전체적으로 작게
      const big   = damage >= 8 && !isGuard;         // 어퍼컷급 — 더 크게 터진다
      const k     = mag * (big ? 1.25 : 1.0);

      // 1) 백색 코어 섬광 — 아주 짧고 강하게 (폭발의 "번쩍")
      push(addSprite(texGlow, 0xffffff, pos, 3 * k, 1), 0.16, 'grow',
           { from: 3 * k, to: 15 * k, fade: 1.3 });

      // 2) 화염구 — 코어보다 크고 조금 길게 남는다
      push(addSprite(texGlow, hot, pos, 2 * k, 1), 0.34, 'grow',
           { from: 2 * k, to: 19 * k, fade: 1.8 });

      // 3) 만화풍 스타버스트 — 회전하며 퍼진다. "폭탄 터지듯"의 핵심
      push(addSprite(texStar, 0xffffff, pos, 1 * k, 1, THREE.AdditiveBlending,
                     Math.random() * Math.PI), 0.32, 'grow',
           { from: 2 * k, to: 21 * k, fade: 1.6, spin: (Math.random() < 0.5 ? -1 : 1) * 2.2 });

      // 4) 충격파 링 — 얇고 빠른 것과 두껍고 느린 것 두 겹. 속도가 달라야 파문처럼 보인다.
      push(addSprite(texRingThin, 0xffffff, pos, 2 * k, 0.95), 0.36, 'grow',
           { from: 2 * k, to: 30 * k, fade: 2.2 });
      push(addSprite(texRing, color, pos, 2 * k, 0.9), 0.60, 'grow',
           { from: 2 * k, to: 19 * k, fade: 2.0 });

      // 5) 연기 — 천천히 부풀며 위로 떠오른다. 폭발에 부피감을 준다.
      push(addSprite(texSmoke, 0x8899aa, pos, 3 * k, 0.5, THREE.NormalBlending),
           0.95, 'grow', { from: 3 * k, to: 20 * k, fade: 1.5, drift: 3.2 });

      // 6) 불꽃 파편
      const s = addSparks(pos, color, isGuard ? 12 : (big ? 40 : 30), isGuard ? 8 : 16, 1.0 * k);
      push(s.pts, isGuard ? 0.5 : 0.75, 'sparks', { vel: s.vel });

      // 7) 임팩트 텍스트 — HIT! / CRITICAL! / BLOCK!
      const label = isGuard ? 'BLOCK!' : (big ? 'CRITICAL!' : 'HIT!');
      const labelCol = isGuard ? '#5fd8ff' : (big ? '#ffd23a' : hexToCss(colorHex));
      const labelH = isGuard ? 3.0 : (big ? 5.2 : 4.2);
      addImpactText(pos.clone().add(new THREE.Vector3(0, 2.6, 0)),
                    label, labelCol, 104, labelH, 0.85, 4.2,
                    (Math.random() - 0.5) * 0.28, 0);

      // 8) 데미지 숫자 — 임팩트 텍스트보다 살짝 늦게 떠서 순서가 읽힌다
      addImpactText(pos.clone().add(new THREE.Vector3(0, 1.0, 0)),
                    `-${damage}`, '#ffffff', 78, 2.8, 0.85, 6.0, 0, 0.10);
    }

    /** K.O. — 타격보다 크고 길게. 넘어지는 연출과 겹쳐 재생한다. */
    function spawnKO(pos, colorHex) {
      const hot = mixHex(colorHex, 0xffffff, 0.5);
      push(addSprite(texGlow, 0xffffff, pos, 5, 1), 0.30, 'grow', { from: 5, to: 30, fade: 1.3 });
      push(addSprite(texGlow, hot, pos, 4, 1), 0.55, 'grow', { from: 4, to: 40, fade: 1.7 });
      push(addSprite(texStar, 0xffffff, pos, 3, 1, THREE.AdditiveBlending, Math.random() * Math.PI),
           0.55, 'grow', { from: 3, to: 44, fade: 1.5, spin: 1.6 });
      push(addSprite(texRingThin, 0xffffff, pos, 3, 1), 0.55, 'grow', { from: 3, to: 52, fade: 2.2 });
      push(addSprite(texRing, colorHex, pos, 3, 1), 0.95, 'grow', { from: 3, to: 36, fade: 2.0 });
      push(addSprite(texSmoke, 0x8899aa, pos, 5, 0.6, THREE.NormalBlending),
           1.5, 'grow', { from: 5, to: 34, fade: 1.4, drift: 3.0 });

      const s = addSparks(pos, colorHex, 70, 22, 1.3);
      push(s.pts, 1.3, 'sparks', { vel: s.vel });

      addImpactText(pos.clone().add(new THREE.Vector3(0, 3.4, 0)),
                    'K.O.', hexToCss(colorHex), 130, 8, 1.7, 2.6, 0, 0);
    }

    function dispose(e) {
      scene.remove(e.obj);
      if (e.obj.geometry) e.obj.geometry.dispose();
      if (e.obj.material) e.obj.material.dispose();
      // e.tex는 textCache가 공유하므로 여기서 dispose하지 않는다
    }

    function update(dt) {
      for (let i = live.length - 1; i >= 0; i--) {
        const e = live[i];

        // 등장 지연 (데미지 숫자를 임팩트 텍스트보다 늦게 띄운다)
        if (e.delay > 0) {
          e.delay -= dt;
          if (e.delay > 0) continue;
        }

        e.t += dt;
        const k = Math.min(1, e.t / e.life);   // 0 → 1

        if (e.kind === 'grow') {
          // 처음에 빠르게 퍼지고 끝으로 갈수록 느려짐 (ease-out)
          const ease = 1 - Math.pow(1 - k, 2.2);
          const sc = e.from + (e.to - e.from) * ease;
          e.obj.scale.set(sc, sc, 1);
          e.obj.material.opacity = Math.pow(1 - k, e.fade);
          if (e.spin) e.obj.material.rotation += e.spin * dt;
          if (e.drift) e.obj.position.y += e.drift * dt;
        } else if (e.kind === 'sparks') {
          const pos = e.obj.geometry.attributes.position;
          for (let j = 0; j < e.vel.length; j++) {
            const v = e.vel[j];
            v.y -= 34 * dt;                    // 중력
            v.multiplyScalar(1 - 2.4 * dt);    // 공기 저항
            pos.array[j * 3]     += v.x * dt;
            pos.array[j * 3 + 1] += v.y * dt;
            pos.array[j * 3 + 2] += v.z * dt;
          }
          pos.needsUpdate = true;
          e.obj.material.opacity = 1 - k * k;
        } else if (e.kind === 'text') {
          e.obj.position.y += e.rise * dt * (1 - k * 0.75);   // 처음엔 빠르게 떠오르다 잦아듦
          // 오버슈트 팝: 0 → 1.45배 → 1.0 (0.18초). 튕겨 나오듯 등장한다.
          const P = 0.18;
          let pop;
          if (e.t < P) {
            const u = e.t / P;
            pop = 0.2 + 1.25 * (1 - Math.pow(1 - u, 3));      // 0.2 → 1.45
          } else {
            const u = Math.min(1, (e.t - P) / 0.14);
            pop = 1.45 - 0.45 * u;                            // 1.45 → 1.0
          }
          e.obj.scale.set(e.baseH * e.tex.aspect * pop, e.baseH * pop, 1);
          // 흔들림 — 등장 직후 살짝 떨린다
          const shake = Math.max(0, 1 - e.t / 0.25);
          e.obj.material.rotation = e.tilt + Math.sin(e.t * 55) * 0.05 * shake;
          e.obj.material.opacity = k < 0.55 ? 1 : 1 - (k - 0.55) / 0.45;
        }

        if (k >= 1) { dispose(e); live.splice(i, 1); }
      }
    }

    /** 라운드 리셋 등에서 화면에 남은 이펙트를 즉시 정리한다. */
    function clear() {
      live.forEach(dispose);
      live.length = 0;
    }

    return { spawnHit, spawnKO, update, clear };
  };
})();
