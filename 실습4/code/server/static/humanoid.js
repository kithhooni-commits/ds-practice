/**
 * humanoid.js — 저폴리 관절형 휴머노이드 빌더 + 절차적 애니메이션
 * (Three.js r128 전역 THREE 의존, ES module 아님)
 *
 * 사용법:
 *   const h = window.createHumanoid(0xff3366);
 *   scene.add(h.group);
 *   h.group.position.set(x, 0, z);
 *   h.group.rotation.y = yaw;
 *
 *   // 매 프레임:
 *   h.setAction("RIGHT_CROSS"); // "LEFT_JAB" · "RIGHT_HOOK" · "LEFT_UPPERCUT" · "DUAL_GUARD" · "IDLE" ...
 *   h.update();                 // idle 바운스/보행/펀치/가드/피격/다운 포즈를 자동 보간
 *
 *   h.hit(damage)   // 피격 리액션 (움찔 + 뒤로 밀림)
 *   h.setDown(true) // K.O. — 뒤로 넘어지며 사라짐 (false면 다시 일어남)
 *   h.setFace(face) // 3D 복원 얼굴을 머리로 사용 (face3d.js 의 createFace3D 결과)
 *
 * 노출 API (기존 코드 호환):
 *   .group, .head, .body(몸통), .leftGlove, .rightGlove, .shield
 *   .armL { shoulder, elbow, glove }, .armR { ... }
 *   .legL { hip, knee }, .legR { hip, knee }
 *   .setAction(action), .update(), .hit(dmg), .setDown(bool), .isDown()
 *
 * 좌표 규약: 아바타 로컬 +z가 정면. 펀치는 +z로 뻗고, 넘어질 때는 -z(뒤)로 눕는다.
 */
(function () {
  if (typeof THREE === 'undefined') {
    console.error('[humanoid.js] THREE 로드 필요 (three.min.js 이후에 include)');
    return;
  }

  /**
   * 캡슐(원통 + 반구 캡). r128 에는 CapsuleGeometry 가 없어서 직접 만든다.
   * 팔다리를 민민한 원통으로 두면 관절이 끊겨 보여 사람으로 읽히지 않는다 —
   * 끝을 둥글게 막아야 근육 덩어리처럼 이어진다.
   * @param {number} rTop  위쪽 반지름 (어깨/허벅지 쪽이 굵다)
   * @param {number} rBot  아래쪽 반지름
   * @param {number} len   원통 길이 (캡 제외). 중심이 원점, +y 위쪽.
   */
  function capsule(rTop, rBot, len, mat, seg) {
    const g = new THREE.Group();
    const n = seg || 10;
    const body = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, len, n), mat);
    g.add(body);
    const top = new THREE.Mesh(new THREE.SphereGeometry(rTop, n, Math.max(6, n >> 1)), mat);
    top.position.y = len / 2;
    top.scale.y = 0.85;
    g.add(top);
    const bot = new THREE.Mesh(new THREE.SphereGeometry(rBot, n, Math.max(6, n >> 1)), mat);
    bot.position.y = -len / 2;
    bot.scale.y = 0.85;
    g.add(bot);
    return g;
  }

  function makeLimb(side) {
    // 팔: 어깨 → 상완 → 팔꿈치 → 전완 → 글러브
    const shoulder = new THREE.Group();
    const elbow = new THREE.Group();
    const upper = capsule(0.30, 0.24, UPPER_LEN - 0.30, null);
    const fore = capsule(0.23, 0.19, FORE_LEN - 0.24, null);
    const glove = new THREE.Mesh(new THREE.SphereGeometry(GLOVE_R, 16, 14), null);
    glove.scale.set(1.0, 1.12, 1.18);   // 복싱 글러브는 앞뒤로 길쭉하다
    return { shoulder, elbow, upper, fore, glove };
  }

  // 액션 이름 → 펀치 종류. 아바타 모션이 기술마다 달라지는 근거표.
  const PUNCH_KIND = {
    LEFT_JAB:       { kind: 'straight', side: 'left'  },
    RIGHT_CROSS:    { kind: 'straight', side: 'right' },
    JAB_STRAIGHT:   { kind: 'straight', side: 'right' },
    LEFT_HOOK:      { kind: 'hook',     side: 'left'  },
    RIGHT_HOOK:     { kind: 'hook',     side: 'right' },
    LEFT_UPPERCUT:  { kind: 'uppercut', side: 'left'  },
    RIGHT_UPPERCUT: { kind: 'uppercut', side: 'right' },
    ENERGY_WAVE:    { kind: 'wave',     side: 'both'  },
  };

  // 기술별 모션 길이(초) — 훅/어퍼는 궤적이 커서 조금 길게 잡아야 눈에 읽힌다.
  const PUNCH_DUR = { straight: 0.30, hook: 0.42, uppercut: 0.42, wave: 0.55 };

  // ── 신체 비율 ─────────────────────────────────────────────────────────
  // 예전 아바타는 머리가 어깨폭만 했고(반지름 1.3) 다리가 키의 24%뿐이라
  // 사람이 아니라 눈사람처럼 보였다. 사람 비율에 가깝게 다시 잡는다.
  //   키 7.2 / 다리 3.2(44%) / 어깨폭 2.1 / 머리 지름 1.44(키의 1/5)
  // 머리를 완전한 실제 비율(1/7.5)로 줄이면 3D 복원 얼굴이 너무 작아 안 보이므로
  // 게임에서 흔히 쓰는 조금 큰 머리로 타협한다.
  const HIP_Y = 3.20;        // 골반
  const CHEST_Y = 4.95;      // 가슴
  const SHOULDER_Y = 5.22;   // 어깨 관절
  const NECK_Y = 5.62;       // 목
  const HEAD_Y = 6.42;       // 머리 중심
  const SHOULDER_X = 1.06;   // 어깨 관절 좌우 위치
  const HIP_X = 0.44;
  const UPPER_LEN = 1.42;    // 상완
  const FORE_LEN = 1.30;     // 전완
  const GLOVE_R = 0.46;
  const THIGH_LEN = 1.55;
  const SHIN_LEN = 1.45;

  // 3D 얼굴을 씌울 때의 두개골 형상.
  // z를 납작하게 눌러 "뒤통수"만 담당하게 하고, 그 앞을 얼굴이 덮는다.
  const HEAD_R = 0.72;                                   // 구 머리 반지름
  const SKULL_SCALE = { x: 0.98, y: 1.06, z: 0.72 };
  const FACE_MARGIN = 0.04;   // 두개골 표면에서 얼굴을 이만큼 더 띄운다 (z-fighting 방지)

  /**
   * 얼굴 메쉬가 두개골 타원체에 **한 정점도 파묻히지 않는** 최소 전방 오프셋을 구한다.
   *
   * bounds.zMin 하나로 어림하면 얼굴 형태에 따라 뚫린다 —
   * 가장 뒤쪽 정점이 반드시 가장 깊이 박히는 정점은 아니기 때문이다(옆으로 벌어진 뺨은
   * 타원체가 좁아지는 자리라 오히려 여유가 있고, 가운데 낮은 정점이 더 위험하다).
   * 정점마다 정확히 풀어서 최댓값을 취한다.
   *
   *   타원체 밖 조건: (x/a)^2 + (y/b)^2 + ((z+d)/c)^2 >= 1
   *   k = 1 - (x/a)^2 - (y/b)^2  가 0 이하면 그 정점은 x·y 만으로 이미 바깥 → 제약 없음
   *   그렇지 않으면  d >= c*sqrt(k) - z
   */
  function faceForwardOffset(face) {
    const a = HEAD_R * SKULL_SCALE.x, b = HEAD_R * SKULL_SCALE.y, c = HEAD_R * SKULL_SCALE.z;
    const attr = face.mesh && face.mesh.geometry && face.mesh.geometry.attributes
              && face.mesh.geometry.attributes.position;
    const pos = attr && attr.array;
    // **반드시 적용된 스케일을 반영해야 한다.** 머리 크기에 맞추려고 메쉬를 줄여 놓고
    // 오프셋만 원본 정점으로 계산하면 얼굴이 엉뚱한 자리에 놓인다(파묻히거나 붕 뜬다).
    // 이름 주의: 아래 루프에 이미 k(타원체 여유분)가 있다. 같은 이름을 쓰면 TDZ 로 죽는다.
    const sc = (face.mesh && face.mesh.scale && face.mesh.scale.x) || 1;
    if (!pos || !pos.length) {
      const zMin = ((face.bounds && face.bounds.zMin) || -0.4) * sc;
      return c - zMin + FACE_MARGIN;      // 바운딩만 있을 때의 안전한 폴백
    }
    let d = -Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      const x = pos[i] * sc, y = pos[i + 1] * sc, z = pos[i + 2] * sc;
      const k = 1 - (x / a) * (x / a) - (y / b) * (y / b);
      if (k <= 0) continue;               // x·y 만으로 이미 타원체 바깥
      const need = c * Math.sqrt(k) - z;
      if (need > d) d = need;
    }
    if (d === -Infinity) d = 0;           // 모든 정점이 이미 바깥
    return d + FACE_MARGIN;
  }

  // 불꽃 오라용 텍스처 — 위로 갈수록 흩어지는 세로 그라디언트.
  // 한 번만 만들어 모든 아바타가 공유한다.
  let auraTex = null;
  function makeAuraTexture() {
    const c = document.createElement('canvas');
    c.width = 64; c.height = 128;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 128, 0, 0);
    grad.addColorStop(0.00, 'rgba(255,240,180,0.95)');   // 뿌리는 밝은 노랑
    grad.addColorStop(0.28, 'rgba(255,150,40,0.80)');
    grad.addColorStop(0.62, 'rgba(255,60,20,0.42)');
    grad.addColorStop(1.00, 'rgba(120,0,0,0)');          // 끝은 흩어진다
    g.fillStyle = grad;
    // 가장자리를 물결지게 깎아 불꽃 혀 모양을 만든다
    g.beginPath();
    g.moveTo(32, 128);
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      const w = 30 * (1 - t) * (0.55 + 0.45 * Math.abs(Math.sin(t * 9)));
      g.lineTo(32 + w, 128 - t * 128);
    }
    for (let i = 20; i >= 0; i--) {
      const t = i / 20;
      const w = 30 * (1 - t) * (0.55 + 0.45 * Math.abs(Math.cos(t * 7)));
      g.lineTo(32 - w, 128 - t * 128);
    }
    g.closePath();
    g.fill();
    return new THREE.CanvasTexture(c);
  }

  window.createHumanoid = function (hexColor, opts) {
    opts = opts || {};
    const color = (hexColor !== undefined && hexColor !== null) ? hexColor : 0xff3366;

    // group = 외부 변환(위치·yaw). rig = 내부 변환(넘어짐·피격 리액션·어퍼컷 상하).
    // 넘어짐을 group에 직접 걸면 호출자가 매 프레임 쓰는 rotation.y와 축이 섞여
    // 바라보는 방향과 무관하게 이상한 쪽으로 눕는다. 그래서 한 겹 분리한다.
    const group = new THREE.Group();
    const rig = new THREE.Group();
    group.add(rig);

    // 재질. metalness 를 0 에 가깝게, roughness 를 높게 잡아야 플라스틱 인형처럼 보이지 않는다.
    // 피부는 아주 옅은 붉은 emissive 를 얹어 빛이 살을 통과하는 느낌(서브서피스 흉내)을 준다.
    const outfitMat = new THREE.MeshStandardMaterial({ color: 0x232839, roughness: 0.88, metalness: 0.04 });
    const skinMat   = new THREE.MeshStandardMaterial({
      color: 0xc08a63, roughness: 0.82, metalness: 0.0,
      emissive: 0x2a0d06, emissiveIntensity: 0.42,
    });
    const accentMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.4, metalness: 0.3 });
    const gloveMat  = new THREE.MeshStandardMaterial({ color: color, emissive: color, emissiveIntensity: 0.35 });
    const visorMat  = new THREE.MeshBasicMaterial({ color: color });
    const fadeMats  = [outfitMat, skinMat, accentMat, gloveMat, visorMat];   // headMat 은 아래에서 추가

    // ---------- 몸통 (Lathe 실루엣) ----------
    // 원통 하나로는 사람 몸통이 되지 않는다. 어깨는 넓고 허리는 잘록한 **윤곽선을 돌려서**
    // 만들고, z 를 눌러 단면을 타원으로 바꾼다 (사람 몸통은 좌우가 앞뒤보다 넓다).
    const torsoProfile = [
      [0.60, HIP_Y - 0.10],
      [0.70, HIP_Y + 0.30],
      [0.66, HIP_Y + 0.72],      // 허리 — 잘록하게
      [0.78, HIP_Y + 1.12],
      [0.94, CHEST_Y],           // 가슴
      [1.03, SHOULDER_Y],        // 어깨
      [0.86, SHOULDER_Y + 0.26],
      [0.40, NECK_Y - 0.04],
    ].map(v => new THREE.Vector2(v[0], v[1]));
    const torso = new THREE.Mesh(new THREE.LatheGeometry(torsoProfile, 16), outfitMat);
    torso.scale.z = 0.70;
    rig.add(torso);

    // 가슴 근육 덩어리 — 정면 실루엣에 두께를 준다
    const pecs = new THREE.Mesh(new THREE.SphereGeometry(0.62, 12, 10), outfitMat);
    pecs.position.set(0, CHEST_Y + 0.05, 0.40);
    pecs.scale.set(1.45, 0.72, 0.55);
    rig.add(pecs);

    // ---------- 골반 / 복싱 트렁크 ----------
    const pelvis = new THREE.Mesh(new THREE.SphereGeometry(0.68, 12, 10), outfitMat);
    pelvis.position.y = HIP_Y;
    pelvis.scale.set(1.05, 0.78, 0.72);
    rig.add(pelvis);

    // 트렁크(반바지) — 밑단이 벌어진 원뿔대. 실루엣이 확 복서다워진다.
    const trunks = new THREE.Mesh(new THREE.CylinderGeometry(0.74, 0.92, 1.15, 14, 1, true), accentMat);
    trunks.position.y = HIP_Y - 0.18;
    trunks.scale.z = 0.78;
    rig.add(trunks);

    // 허리 밴드
    const belt = new THREE.Mesh(new THREE.CylinderGeometry(0.76, 0.76, 0.26, 14), gloveMat);
    belt.position.y = HIP_Y + 0.42;
    belt.scale.z = 0.76;
    rig.add(belt);

    // ---------- 목 / 머리 ----------
    const neck = capsule(0.26, 0.30, 0.34, skinMat, 10);
    neck.position.y = NECK_Y;
    rig.add(neck);

    // 머리는 전용 머티리얼을 쓴다 — accentMat 을 공유하면 3D 얼굴을 씌울 때
    // 발/트렁크 등 accentMat 을 쓰는 다른 부위 색까지 같이 바뀐다.
    const headMat = new THREE.MeshStandardMaterial({
      color: 0xc08a63, roughness: 0.82, metalness: 0.0,
      emissive: 0x2a0d06, emissiveIntensity: 0.42,
    });
    const head = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R, 20, 16), headMat);
    head.position.y = HEAD_Y;
    head.scale.set(0.94, 1.10, 0.96);      // 사람 머리는 위아래로 길다
    rig.add(head);

    fadeMats.push(headMat);

    // 턱 — 구만 있으면 인형 머리다. 아래쪽에 턱 덩어리를 붙여 실루엣을 만든다.
    const jaw = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R * 0.78, 14, 12), headMat);
    jaw.position.set(0, HEAD_Y - HEAD_R * 0.52, HEAD_R * 0.16);
    jaw.scale.set(0.94, 0.72, 1.02);
    rig.add(jaw);

    // 귀
    const ears = [];
    [-1, 1].forEach(side => {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(HEAD_R * 0.24, 10, 8), headMat);
      ear.position.set(side * HEAD_R * 0.92, HEAD_Y + 0.02, -HEAD_R * 0.06);
      ear.scale.set(0.45, 1.0, 0.75);
      rig.add(ear);
      ears.push(ear);
    });

    // 헤드기어 — 파이터 색을 식별할 수 있게 머리에 띠를 두른다
    const headgear = new THREE.Mesh(new THREE.TorusGeometry(HEAD_R * 0.92, 0.11, 8, 20), accentMat);
    headgear.position.set(0, HEAD_Y + 0.22, 0);
    headgear.rotation.x = Math.PI / 2;
    headgear.scale.set(1.0, 1.0, 1.15);
    rig.add(headgear);

    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.86, 0.20, 0.32), visorMat);
    visor.position.set(0, HEAD_Y + 0.02, HEAD_R * 0.92);
    rig.add(visor);

    // ---------- 팔 (어깨/팔꿈치 관절) ----------
    const armL = makeLimb(-1), armR = makeLimb(1);

    [armL, armR].forEach(arm => {
      const side = (arm === armL) ? -1 : 1;
      arm.shoulder.position.set(side * SHOULDER_X, SHOULDER_Y, 0);
      arm.upper.children.forEach(m => { m.material = skinMat; });
      arm.upper.position.y = -UPPER_LEN / 2;
      arm.fore.children.forEach(m => { m.material = skinMat; });
      arm.fore.position.y = -FORE_LEN / 2;
      arm.glove.material = gloveMat;
      arm.glove.position.y = -FORE_LEN - GLOVE_R * 0.45;
      arm.elbow.position.y = -UPPER_LEN;

      // 어깨 근육 — 팔이 몸통에서 매끄럽게 이어지게 덮는다
      const delt = new THREE.Mesh(new THREE.SphereGeometry(0.36, 12, 10), skinMat);
      delt.scale.set(1.0, 0.95, 1.0);
      arm.shoulder.add(delt);

      arm.shoulder.add(arm.upper);
      arm.shoulder.add(arm.elbow);
      arm.elbow.add(arm.fore);
      arm.elbow.add(arm.glove);
      rig.add(arm.shoulder);
    });

    // ---------- 다리 (고관절/무릎 관절) ----------
    function makeLeg() {
      const hip = new THREE.Group();
      const knee = new THREE.Group();
      const thigh = capsule(0.40, 0.31, THIGH_LEN - 0.40, outfitMat);
      const shin = capsule(0.29, 0.20, SHIN_LEN - 0.30, skinMat);
      // 복싱화 — 발목까지 올라오는 신발
      const boot = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.27, 0.62, 10), accentMat);
      shaft.position.y = 0.22;
      const sole = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.20, 0.86), outfitMat);
      sole.position.set(0, -0.10, 0.16);
      boot.add(shaft); boot.add(sole);
      return { hip, knee, thigh, shin, foot: boot };
    }
    const legL = makeLeg(), legR = makeLeg();
    [legL, legR].forEach(leg => {
      const side = leg === legL ? -1 : 1;
      leg.hip.position.set(side * HIP_X, HIP_Y - 0.20, 0);
      leg.thigh.position.y = -THIGH_LEN / 2;
      leg.knee.position.y = -THIGH_LEN;
      leg.shin.position.y = -SHIN_LEN / 2;
      leg.foot.position.set(0, -SHIN_LEN - 0.10, 0.06);
      leg.hip.add(leg.thigh);
      leg.hip.add(leg.knee);
      leg.knee.add(leg.shin);
      leg.knee.add(leg.foot);
      rig.add(leg.hip);
    });

    // ---------- 복싱 스탠스 ----------
    // 정면으로 뻣뻣하게 선 자세는 복서로 보이지 않는다.
    // 몸을 살짝 비스듬히 틀고(bladed) 앞발을 내밀며 무릎을 굽힌다.
    // 회전은 rig 가 아니라 부위별로 준다 — rig.rotation 은 피격·K.O. 연출이 쓰기 때문이다.
    const STANCE_TWIST = 0.30;                 // 몸통 비틀기 (약 17도)
    torso.rotation.y = STANCE_TWIST;
    pecs.rotation.y = STANCE_TWIST;
    trunks.rotation.y = STANCE_TWIST;
    legL.hip.position.z = 0.42;                // 왼발이 앞
    legR.hip.position.z = -0.40;               // 오른발이 뒤
    legL.hip.rotation.y = 0.12;
    legR.hip.rotation.y = -0.50;               // 뒷발은 바깥으로 벌린다
    legL.foot.rotation.y = -0.10;
    legR.foot.rotation.y = 0.24;

    // ---------- 가드 실드 (홀로그램) ----------
    const shield = new THREE.Mesh(
      new THREE.SphereGeometry(1.9, 20, 16, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: color, wireframe: true, transparent: true, opacity: 0 })
    );
    shield.rotation.x = Math.PI / 2;
    shield.position.y = CHEST_Y + 0.1;
    rig.add(shield);

    // ---------- 분노 오라 (불꽃) ----------
    // 분노 게이지가 차오르면 몸에서 불길이 인다. 게이지가 가득 차면 필살기를 쓸 수 있다.
    // 빌보드 스프라이트를 몸 둘레에 여러 장 세워 돌린다 — 어느 각도에서 봐도 불꽃으로 보인다.
    if (!auraTex) auraTex = makeAuraTexture();
    const aura = new THREE.Group();
    const auraFlames = [];
    const AURA_N = 9;
    for (let i = 0; i < AURA_N; i++) {
      const m = new THREE.SpriteMaterial({
        map: auraTex, color: 0xff7a20, transparent: true, opacity: 0,
        blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const sp = new THREE.Sprite(m);
      const a = (i / AURA_N) * Math.PI * 2;
      const r = 0.95 + (i % 3) * 0.18;
      sp.position.set(Math.cos(a) * r, HIP_Y + 0.4, Math.sin(a) * r);
      sp.scale.set(1.5, 2.6, 1);
      aura.add(sp);
      auraFlames.push({ sp, a, r, phase: Math.random() * Math.PI * 2,
                        baseY: HIP_Y + 0.2 + (i % 4) * 0.35 });
    }
    // 발밑 불꽃 고리 — 서 있는 자리에서 타오르는 느낌
    const auraRingMat = new THREE.SpriteMaterial({
      map: auraTex, color: 0xffb040, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, rotation: 0,
    });
    const auraRing = new THREE.Sprite(auraRingMat);
    auraRing.position.y = 0.35;
    auraRing.scale.set(4.2, 1.6, 1);
    aura.add(auraRing);
    rig.add(aura);

    // ---------- 애니메이션 상태 ----------
    const S = {
      action: 'IDLE',
      punch: 0,          // 1 → 0 감쇠 (남은 비율)
      punchSide: 'right',
      punchKind: 'straight',
      punchDur: PUNCH_DUR.straight,
      guard: 0,          // 0..1 (감쇠)
      flinch: 0,         // 피격 움찔 0..1
      flinchMag: 1,      // 데미지에 비례한 세기
      down: false,       // K.O. 여부
      downAmt: 0,        // 넘어짐 진행도 0..1
      rage: 0,           // 분노 게이지 0..1 (서버 값 / 100)
      rageNow: 0,        // 표시용 평활값
      walkPhase: 0,
      lastPos: new THREE.Vector3(),
      lastT: performance.now(),
      // 현재 포즈 (lerp 대상으로 점진 수렴) — 시작부터 가드 자세
      lSX: -1.05, lSZ: -0.30, lEX: -2.10,
      rSX: -1.05, rSZ: 0.30, rEX: -2.10
    };

    // 정지 포즈.
    //
    // 예전 neutral 은 팔을 거의 늘어뜨린 자세(어깨 0.15)였다. 복서는 **항상 손을 올리고 있다** —
    // 팔을 내린 채 서 있으면 아무리 몸을 잘 만들어도 복서로 보이지 않는다.
    // 그래서 기본 자세부터 가드로 잡고, guard 는 거기서 한 단계 더 조인 형태로 둔다.
    const POSES = {
      // 스탠스: 주먹을 턱 높이로, 팔꿈치는 몸통에 붙인다
      neutral: { lSX: -1.05, lSZ: -0.30, lEX: -2.10, rSX: -1.05, rSZ: 0.30, rEX: -2.10 },
      // 가드: 더 높고 더 좁게 — 얼굴을 완전히 가린다
      guard:   { lSX: -1.30, lSZ: -0.16, lEX: -2.42, rSX: -1.30, rSZ: 0.16, rEX: -2.42 }
    };

    /**
     * 기술별 팔 포즈. 이 함수가 "쨉/훅/어퍼가 서로 다르게 보이는" 실체다.
     *   p — 모션 진행도 0(시작) → 1(끝). 단조 증가. 궤적을 한 방향으로 쓸 때 쓴다.
     *   w — 뻗음 정도 0 → 1 → 0. 뻗었다가 회수하는 왕복에 쓴다.
     *   s — 좌우 부호 (왼팔 -1 / 오른팔 +1). 어깨 z회전이 몸 바깥쪽으로 향하는 방향.
     * 반환: SX(어깨 앞뒤) · SZ(어깨 좌우) · EX(팔꿈치 굽힘) · twist(몸통 회전) · lift(상하)
     */
    function punchPose(kind, s, p, w) {
      if (kind === 'hook') {
        // 훅: 팔꿈치를 90°로 접은 채 바깥에서 안쪽으로 감아친다.
        // 스윙(SZ)은 p로 한 방향으로만 쓸어야 "휘두른다"로 읽힌다. w로 하면 갔다 되돌아온다.
        return {
          SX: -1.15 - 0.30 * w,          // 어깨 높이로 들어올린 채
          SZ: s * (1.25 - 1.60 * p),     // 바깥(1.25) → 몸 앞을 가로질러(-0.35)
          EX: -1.70 + 0.20 * w,          // 팔꿈치는 끝까지 접힌 채 유지
          twist: -s * 0.55 * w,          // 훅은 몸통 회전이 주력 — 크게 돌린다
          lift: 0
        };
      }
      if (kind === 'uppercut') {
        // 어퍼컷: 무릎을 낮췄다가 아래에서 위로 솟구친다.
        return {
          SX: -0.35 - 1.75 * w,          // 아래로 내렸다(-0.35) → 위 앞(-2.10)
          SZ: s * (0.38 - 0.18 * w),
          EX: -2.35 + 0.55 * w,          // 깊게 접은 팔꿈치
          twist: -s * 0.30 * w,
          lift: -0.40 * Math.sin(p * Math.PI * 0.9) + 0.60 * w  // 살짝 가라앉았다 솟음
        };
      }
      if (kind === 'wave') {
        // 필살기: **모았다가 밀어낸다**. 한 번에 뻗으면 그냥 양손 스트레이트로 보인다.
        //   0 ~ 0.34 : 웅크리며 양손을 가슴에 모은다 (기 모으기)
        //   0.34 ~ 1 : 몸을 펴며 양손을 앞으로 쏘아 보낸다
        const GATHER = 0.34;
        if (p < GATHER) {
          const g = p / GATHER;                       // 0 → 1
          return {
            SX: -1.05 - 0.42 * g,                     // 팔꿈치를 몸에 붙이며
            SZ: s * (0.45 - 0.34 * g),                // 손을 가운데로 모은다
            EX: -2.10 - 0.55 * g,                     // 더 깊게 접는다
            twist: 0,
            lift: -0.55 * g                           // 웅크린다
          };
        }
        const q = (p - GATHER) / (1 - GATHER);        // 0 → 1
        const push = Math.sin(Math.min(1, q * 1.35) * Math.PI * 0.5);   // 앞으로 쏘는 정도
        return {
          SX: -1.47 + 0.10 * push,                    // 정면 수평으로
          SZ: s * (0.11 + 0.12 * push),               // 두 손을 나란히
          EX: -2.65 + 2.55 * push,                    // 깊게 접었던 팔을 한 번에 편다
          twist: 0,
          lift: -0.55 + 1.25 * push                   // 웅크렸다 솟구친다
        };
      }
      // 쨉 / 스트레이트: 가드에서 팔꿈치를 펴면서 정면으로 곧게 뻗는다.
      return {
        SX: -1.05 - 0.52 * w,      // 가드(-1.05) → 정면 수평(-1.57)
        SZ: s * (0.30 - 0.16 * w),
        EX: -2.10 + 2.02 * w,      // 접힌 팔꿈치 → 곧게 폄
        twist: -s * 0.28 * w,
        lift: 0
      };
    }

    function setAction(action) {
      if (!action) return;
      S.action = action;
      const spec = PUNCH_KIND[action];
      if (spec) {
        S.punch = 1;
        S.punchKind = spec.kind;
        S.punchSide = spec.side;
        S.punchDur = PUNCH_DUR[spec.kind] || 0.30;
      } else if (action === 'DUAL_GUARD' || action === 'TWO_HAND_GUARD') {
        S.guard = 1;
      }
    }

    /**
     * 3D 복원 얼굴을 머리에 붙인다 (face3d.js).
     * 단색 구 머리는 숨기고, 얼굴 메쉬를 같은 자리에 놓는다.
     * face 를 null 로 주면 원래 구 머리로 되돌아간다.
     */
    let faceObj = null;
    function setFace(face) {
      if (faceObj && faceObj.mesh && faceObj.mesh.parent) rig.remove(faceObj.mesh);
      faceObj = face || null;

      if (faceObj && faceObj.mesh) {
        // **피부톤을 얼굴 사진에 맞춘다.** 목·팔다리가 원래 색 그대로면 목선에서 색이 끊긴다.
        if (faceObj.skinTone) {
          headMat.color.setHex(faceObj.skinTone);
          skinMat.color.setHex(faceObj.skinTone);
        }

        const fb = faceObj.bounds;
        const fullHead = !!faceObj.isFullHead;

        // 머리 크기에 맞춘다. 호출부가 어떤 width 로 만들었든 여기서 실측 폭으로 다시 맞춘다.
        let k = 1;
        if (fb && fb.xMax > fb.xMin) {
          k = HEAD_R * SKULL_SCALE.x * (fullHead ? 2.20 : 2.05) / (fb.xMax - fb.xMin);
          faceObj.mesh.scale.setScalar(k);
        }

        if (fullHead) {
          // 뒤통수까지 닫힌 머리다 — 구·턱·바이저·헤드기어를 **전부** 숨긴다.
          // 앞뒤로 겹치는 것이 없으니 "가면을 쓴 인형"이 될 여지가 사라진다.
          head.visible = false;
          jaw.visible = false;
          visor.visible = false;
          headgear.visible = false;
          // 메쉬 중심을 머리 위치에 맞춘다. 얼굴만 있을 때와 달리 두개골 때문에
          // 무게중심이 뒤로 가 있으므로 실제 바운딩 중심을 빼 준다.
          faceObj.__yOff = -(fb.yMin + fb.yMax) / 2 * k;
          faceObj.mesh.position.set(
            -(fb.xMin + fb.xMax) / 2 * k,
            head.position.y + faceObj.__yOff,
            -(fb.zMin + fb.zMax) / 2 * k
          );
          // 귀는 머리 옆면에 맞춰 다시 놓는다
          const halfW = (fb.xMax - fb.xMin) / 2 * k;
          ears.forEach((e, i) => {
            e.position.x = (i === 0 ? -1 : 1) * halfW * 0.94;
            e.visible = true;
          });
        } else {
          // 앞면만 있는 메쉬(폴백) — 구를 두개골로 남기고 그 **앞면 바깥**에 얹는다.
          // 배치를 상수로 박으면 안 된다. 얼굴 깊이는 사람마다 다르고, 조금만 뒤로 가면
          // 얼굴 전체가 구 안에 파묻혀 아무것도 안 보인다(실제로 그렇게 됐었다).
          head.scale.set(SKULL_SCALE.x, SKULL_SCALE.y, SKULL_SCALE.z);
          faceObj.__yOff = 0;
          faceObj.mesh.position.set(0, head.position.y, faceForwardOffset(faceObj));
          head.visible = true;
          jaw.visible = true;
          visor.visible = false;
          headgear.visible = false;
        }
        rig.add(faceObj.mesh);
      } else {
        head.visible = true;
        jaw.visible = true;
        head.scale.set(0.94, 1.10, 0.96);
        headMat.color.setHex(0xc08a63);
        visor.visible = true;
        headgear.visible = true;
        ears.forEach((e, i) => { e.position.x = (i === 0 ? -1 : 1) * HEAD_R * 0.92; });
      }
    }

    /** 피격 리액션 — 데미지가 클수록 크게 움찔한다. */
    function hit(damage) {
      if (faceObj) {
        // 어느 쪽을 맞았는지는 알 수 없으므로 번갈아 — 같은 자리만 계속 눌리면 부자연스럽다
        const sides = ['left', 'right', 'center', 'chin'];
        faceObj.hit(damage, sides[(S.hitCount = (S.hitCount || 0) + 1) % sides.length]);
      }
      S.flinch = 1;
      S.flinchMag = Math.min(1.4, 0.45 + (damage || 5) / 12);
    }

    /**
     * 분노 게이지 반영 (0~100). 차오를수록 불꽃이 커지고, 가득 차면 최대로 타오른다.
     * 게이지가 눈에 보여야 "지금 필살기를 쓸 수 있다"를 조작 없이 알 수 있다.
     */
    function setRage(v) {
      S.rage = Math.max(0, Math.min(1, (v || 0) / 100));
    }

    /** K.O. — true면 뒤로 넘어지며 페이드아웃, false면 되살아난다. */
    function setDown(isDown) {
      S.down = !!isDown;
      if (!S.down) group.visible = true;
    }

    function update() {
      const now = performance.now();
      let dt = (now - S.lastT) / 1000;
      if (dt > 0.05) dt = 0.05; // 탭 전환 시 점프 방지
      S.lastT = now;

      // ---------- K.O. 다운 (다른 모든 포즈보다 우선) ----------
      const downTarget = S.down ? 1 : 0;
      const downRate = dt / (S.down ? 0.75 : 0.45);   // 넘어지는 건 느리게, 일어나는 건 빠르게
      S.downAmt += Math.max(-downRate, Math.min(downRate, downTarget - S.downAmt));
      S.downAmt = Math.max(0, Math.min(1, S.downAmt));

      // 넘어짐 진행도(downEase)만 여기서 구하고, 실제 변환은 update() 맨 끝에서 적용한다.
      // 앞에서 rig를 건드리면 아래의 피격·어퍼컷 리액션이 그대로 덮어써 자세가 튄다.
      const downEase = S.downAmt * S.downAmt * (3 - 2 * S.downAmt);  // smoothstep
      if (S.downAmt > 0) {
        const op = 1 - downEase;
        fadeMats.forEach(m => { m.transparent = true; m.opacity = op; });
        group.visible = op > 0.02;
        if (S.downAmt >= 1) {            // 완전히 쓰러지면 나머지 연출은 계산할 필요가 없다
          shield.material.opacity = 0;
          return;
        }
      } else if (fadeMats[0].opacity !== 1) {
        fadeMats.forEach(m => { m.opacity = 1; m.transparent = false; });
      }

      // 이동 속도 감지 (group 위치 변화)
      const dx = group.position.x - S.lastPos.x;
      const dz = group.position.z - S.lastPos.z;
      const speed = Math.hypot(dx, dz) / Math.max(dt, 0.001);
      S.lastPos.set(group.position.x, group.position.y, group.position.z);
      if (speed > 0.4) S.walkPhase += dt * speed * 1.3;
      else S.walkPhase *= 0.9;

      // 감쇠는 프레임 수가 아니라 시간 기준 (렌더 FPS가 달라도 같은 길이로 보이도록)
      S.punch  = Math.max(0, S.punch - dt / S.punchDur);
      S.guard  = Math.max(0, S.guard - dt / 0.50);   // 가드 ~0.5초 (100ms마다 갱신되므로 유지됨)
      S.flinch = Math.max(0, S.flinch - dt / 0.35);

      // 목표 포즈 결정 — 펀치가 가드보다 우선.
      // 가드를 우선하면, 클라이언트가 10Hz로 보내는 DUAL_GUARD가 S.guard를 계속 1로 되살려
      // 펀치 포즈가 화면에 아예 나타나지 않는다. (복싱 스탠스는 상시 가드 판정)
      const punching = S.punch > 0.02;
      let target, twist = 0, lift = 0;

      if (punching) {
        const p = 1 - S.punch;                                  // 0 → 1
        const w = Math.sin(Math.pow(p, 0.65) * Math.PI);        // 0 → 1 → 0 (빠르게 뻗고 천천히 회수)
        // 치지 않는 팔은 가드를 올린 채로 둔다 — 실제 복싱 폼이고, 어느 팔로 쳤는지가 선명해진다.
        const G = POSES.guard;
        if (S.punchSide === 'both') {
          const pl = punchPose(S.punchKind, -1, p, w);
          const pr = punchPose(S.punchKind,  1, p, w);
          target = { lSX: pl.SX, lSZ: pl.SZ, lEX: pl.EX, rSX: pr.SX, rSZ: pr.SZ, rEX: pr.EX };
          twist = pl.twist; lift = pl.lift;
        } else if (S.punchSide === 'left') {
          const q = punchPose(S.punchKind, -1, p, w);
          target = { lSX: q.SX, lSZ: q.SZ, lEX: q.EX, rSX: G.rSX, rSZ: G.rSZ, rEX: G.rEX };
          twist = q.twist; lift = q.lift;
        } else {
          const q = punchPose(S.punchKind, 1, p, w);
          target = { lSX: G.lSX, lSZ: G.lSZ, lEX: G.lEX, rSX: q.SX, rSZ: q.SZ, rEX: q.EX };
          twist = q.twist; lift = q.lift;
        }
      } else if (S.guard > 0.3) {
        target = POSES.guard;
      } else {
        target = POSES.neutral;
      }

      // 부드러운 lerp — 펀치(공격) 순간엔 빠르게, 회수/대기 시엔 부드럽게
      const k = punching ? 0.55 : 0.16;
      S.lSX += (target.lSX - S.lSX) * k;
      S.lSZ += (target.lSZ - S.lSZ) * k;
      S.lEX += (target.lEX - S.lEX) * k;
      S.rSX += (target.rSX - S.rSX) * k;
      S.rSZ += (target.rSZ - S.rSZ) * k;
      S.rEX += (target.rEX - S.rEX) * k;

      armL.shoulder.rotation.x = S.lSX;
      armL.shoulder.rotation.z = S.lSZ;
      armL.elbow.rotation.x = S.lEX;
      armR.shoulder.rotation.x = S.rSX;
      armR.shoulder.rotation.z = S.rSZ;
      armR.elbow.rotation.x = S.rEX;

      // 피격 리액션 — 뒤로 젖히며 밀린다. 펀치 몸통 회전과 더해진다.
      const fl = S.flinch * S.flinchMag;
      rig.rotation.x = -fl * 0.30;
      rig.rotation.y += (twist - rig.rotation.y) * 0.35;
      rig.position.z = -fl * 0.9;
      rig.position.y = lift + fl * 0.15;
      head.rotation.x = -fl * 0.55;

      // 3D 얼굴 — 호흡/피격/표정 갱신은 얼굴 모듈이 스스로 한다
      if (faceObj) {
        faceObj.update(dt);
        // 호흡에 맞춰 머리와 함께 오르내린다. 닫힌 머리는 바운딩 중심 보정이 들어가 있으므로
        // setFace 가 계산한 오프셋(__yOff)을 유지한 채 y 만 따라가게 한다.
        faceObj.mesh.position.y = head.position.y + (faceObj.__yOff || 0);
        faceObj.mesh.visible = group.visible;
      }

      // 호흡/바운스 — 복서는 늘 리듬을 타고 있다. 미세한 상하 움직임이 "살아있음"을 만든다.
      const breathe = Math.sin(now * 0.004) * 0.05;
      const bob = Math.sin(now * 0.0031) * 0.045;         // 스탠스 바운스
      torso.position.y = breathe + bob;
      pecs.position.y = CHEST_Y + 0.05 + breathe + bob;
      pelvis.position.y = HIP_Y + bob * 0.7;
      trunks.position.y = HIP_Y - 0.18 + bob * 0.7;
      belt.position.y = HIP_Y + 0.42 + bob * 0.75;
      neck.position.y = NECK_Y + breathe * 0.8 + bob;
      head.position.y = HEAD_Y + breathe + bob;
      headgear.position.y = HEAD_Y + 0.22 + breathe + bob;
      jaw.position.y = HEAD_Y - HEAD_R * 0.52 + breathe + bob;
      ears[0].position.y = ears[1].position.y = HEAD_Y + 0.02 + breathe + bob;
      visor.position.y = HEAD_Y + 0.02 + breathe + bob;
      armL.shoulder.position.y = SHOULDER_Y + breathe * 0.9 + bob;
      armR.shoulder.position.y = SHOULDER_Y + breathe * 0.9 + bob;

      // 다리: 이동 시 보행 스윙, 대기 시 미세 자세.
      // 어퍼컷은 다리로 밀어올리는 기술이라 무릎 굽힘을 lift와 연동한다.
      // 다리 — 복싱 스탠스는 **항상 무릎이 굽어 있다**. 곧게 편 다리는 복서로 보이지 않는다.
      const moving = speed > 0.4 ? 1 : 0;
      const swing = Math.sin(S.walkPhase);
      const crouch = (punching && S.punchKind === 'uppercut') ? Math.max(0, -lift) * 0.9 : 0;
      const STANCE_KNEE = 0.34;                      // 기본 무릎 굽힘
      legL.hip.rotation.x = -0.16 + swing * 0.45 * moving + Math.sin(now * 0.003) * 0.02;
      legR.hip.rotation.x = 0.20 - swing * 0.45 * moving + Math.sin(now * 0.003 + Math.PI) * 0.02;
      const kneeBend = moving ? Math.max(0, Math.sin(S.walkPhase + Math.PI) * 0.45) : 0;
      legL.knee.rotation.x = STANCE_KNEE + kneeBend + crouch;
      legR.knee.rotation.x = STANCE_KNEE + 0.10
                           + Math.max(0, Math.sin(S.walkPhase) * 0.45) * moving + crouch;

      // 가드 실드 시각화
      shield.material.opacity += (((S.guard > 0.3 && !punching) ? 0.85 : 0) - shield.material.opacity) * 0.2;

      // ── 분노 오라 ──────────────────────────────────────────────────
      // 게이지는 급변하지만 불꽃은 서서히 붙어야 자연스럽다.
      S.rageNow += (S.rage - S.rageNow) * Math.min(1, dt * 2.5);
      const rg = S.downAmt > 0 ? 0 : S.rageNow;
      aura.visible = rg > 0.02;
      if (aura.visible) {
        const full = rg >= 0.995;                       // 가득 참 = 필살기 사용 가능
        const t2 = now * 0.001;
        for (let i = 0; i < auraFlames.length; i++) {
          const f = auraFlames[i];
          // 몸 둘레를 천천히 돈다 — 정지 화면에서도 살아 있게 보인다
          const ang = f.a + t2 * (full ? 1.5 : 0.8);
          const flick = 0.72 + 0.28 * Math.sin(t2 * 7 + f.phase);
          f.sp.position.set(Math.cos(ang) * f.r * (0.85 + rg * 0.35),
                            f.baseY + Math.sin(t2 * 3.4 + f.phase) * 0.28 + rg * 0.5,
                            Math.sin(ang) * f.r * (0.85 + rg * 0.35));
          f.sp.scale.set(1.15 * rg * flick + 0.35, (2.0 + rg * 2.2) * flick, 1);
          f.sp.material.opacity = rg * (full ? 0.92 : 0.62) * flick;
          // 가득 차면 색이 붉은 주황에서 흰 노랑으로 — "달아올랐다"가 보인다
          f.sp.material.color.setHex(full ? 0xffd23a : 0xff6a18);
        }
        auraRingMat.opacity = rg * 0.55 * (0.75 + 0.25 * Math.sin(t2 * 6));
        auraRing.scale.set(3.4 + rg * 1.6, 1.1 + rg * 0.9, 1);
        auraRingMat.color.setHex(rg >= 0.995 ? 0xffe27a : 0xff8a30);
      }

      // K.O. 다운 변환은 마지막에 덧씌운다 (위의 리액션 값 위에 얹혀 자연스럽게 넘어간다)
      if (downEase > 0) {
        rig.rotation.x -= downEase * 1.48;    // 뒤(-z)로 눕는다
        rig.position.y -= downEase * 2.60;    // 매트로 가라앉음 (키가 커진 만큼 더 내려간다)
        rig.position.z -= downEase * 1.90;
        shield.material.opacity *= (1 - downEase);
      }
    }

    return {
      group, rig, head, jaw, ears, headgear, body: torso,
      leftGlove: armL.glove, rightGlove: armR.glove, shield,
      armL, armR, legL, legR, visor,
      setAction, update, hit, setDown, setFace, setRage,
      getRage: () => S.rage,
      getFace: () => faceObj,
      isDown: () => S.down,
      state: S
    };
  };
})();
