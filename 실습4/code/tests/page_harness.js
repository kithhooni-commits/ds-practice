/**
 * page_harness.js — 실제 브라우저로 페이지를 띄워 런타임 예외를 잡는다.
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd iter3/tests && node page_harness.js [베이스URL]
 *
 * 이 하니스가 따로 존재하는 이유:
 *   `const fx = window.createHitEffects(scene)` 가 animate() 안의 정면 벡터 `const fx` 에
 *   가려져 `fx.update is not a function` 이 매 프레임 터졌고, 그 지점이 renderer.render() 앞이라
 *   **1인칭 화면만 통째로 검은 화면**이 됐다. 이때 로직 하니스(pose/effects/aim/move)는 전부 통과했다.
 *   페이지를 진짜로 띄워보지 않으면 못 잡는 종류의 버그라 경로를 따로 둔다.
 */
const { open, sleep, isNoise } = require('./_cdp');
const BASE = process.argv[2] || 'https://localhost:8000';

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

(async () => {
  console.log('--- Fighter 1인칭 페이지 ---');
  const f = await open(`${BASE}/client?id=client_2`, { port: 9341, settle: 8000 });
  const st = await f.evaluate(`(() => {
    const c = document.querySelector('#first-person-ring canvas');
    const box = document.getElementById('first-person-ring');
    return {
      canvas: !!c, w: c ? c.width : 0, h: c ? c.height : 0,
      boxW: box ? box.clientWidth : 0, boxH: box ? box.clientHeight : 0,
      fps: parseInt(document.getElementById('fps-badge').innerText) || 0,
      animError: document.getElementById('anim-error').style.display,
      hud: document.getElementById('move-hud').innerText
    };
  })()`);
  const errs = f.logs.filter(l => l.kind === 'EXCEPTION');
  const targetLine = (String(st.hud).split('\n').find(l => l.startsWith('Target:')) || '').trim();

  ck('렌더 캔버스가 만들어진다', st.canvas);
  ck('캔버스 크기가 0이 아니다', st.w > 0 && st.h > 0, `${st.w}x${st.h}`);
  ck('컨테이너를 가득 채운다', st.w === st.boxW && st.h === st.boxH, `box ${st.boxW}x${st.boxH}`);
  ck('rAF 루프가 실제로 돈다 (FPS > 0)', st.fps > 0, `${st.fps} FPS`);
  ck('렌더 예외 배너가 뜨지 않는다', st.animError !== 'block');
  ck('런타임 예외 0건', errs.length === 0, errs.map(e => e.text).join(' / ') || 'none');
  ck('HUD에 Target 줄이 있다', /Target:/.test(st.hud));
  ck('혼자 접속이어도 타깃을 잡는다', /Target: P\d/.test(st.hud), targetLine);

  const noisy = f.logs.filter(l => !isNoise(l) && /error/i.test(l.kind));
  ck('예상치 못한 에러 로그 없음', noisy.length === 0, noisy.map(l => l.text).join(' / ') || 'none');

  // 펀치 판정이 정말 punch_core.js 를 통해 도는지 (하니스와 같은 모듈인지) 확인
  const pc = await f.evaluate(`(() => ({
    lib: typeof window.PunchCore,
    factory: typeof (window.PunchCore || {}).createPunchCore,
    instance: typeof punchCore,
    hasTryPunch: typeof (punchCore || {}).tryPunch,
    merged: TUNE.PUNCH_SPEED,
    coreSpeed: punchCore.tune.PUNCH_SPEED,
    armsShared: punchCore.arms === arms
  }))()`);
  ck('punch_core.js 가 로드된다', pc.lib === 'object' && pc.factory === 'function');
  ck('런타임이 판정기 인스턴스를 갖는다', pc.instance === 'object' && pc.hasTryPunch === 'function');
  ck('TUNE 에 펀치 상수가 병합된다', pc.merged === pc.coreSpeed && pc.merged > 0,
     `TUNE.PUNCH_SPEED=${pc.merged}`);
  ck('팔 상태를 코어와 공유한다', pc.armsShared === true);

  console.log('');
  console.log('--- 1인칭 HUD ---');
  // HP 바
  const hp = await f.evaluate(`(() => {
    renderHpBar(100); const full = document.getElementById('hp-fill').style.width;
    renderHpBar(18);
    const low = document.getElementById('hp-fill');
    const r = { full, low: low.style.width, col: low.style.background,
                crit: low.classList.contains('critical'),
                num: document.getElementById('hp-num').innerText };
    renderHpBar(100); return r;
  })()`);
  ck('HP 바가 100에서 가득 찬다', hp.full === '100%', hp.full);
  ck('HP 18이면 폭이 18퍼센트', hp.low === '18%', hp.low);
  ck('숫자도 함께 갱신된다', hp.num === '18', hp.num);
  ck('위험 구간은 붉은색 + 맥박', hp.crit && /255,\s*51,\s*102|#ff3366/i.test(hp.col), hp.col);

  // 미니맵 — 실제로 픽셀이 그려졌는지 본다
  const mini = await f.evaluate(`(() => {
    drawMinimap();
    const c = document.getElementById('minimap');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let painted = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 8) painted++;
    return { w: c.width, h: c.height, painted, total: d.length / 4 };
  })()`);
  ck('미니맵 캔버스가 있다', mini.w > 0 && mini.h > 0, `${mini.w}x${mini.h}`);
  ck('미니맵이 실제로 그려진다', mini.painted > mini.total * 0.3,
     `칠해진 픽셀 ${mini.painted}/${mini.total}`);

  // 내 위치가 미니맵에 반영되는가 — 좌표를 바꾸면 그림이 달라져야 한다
  const moved = await f.evaluate(`(() => {
    const c = document.getElementById('minimap'), g = c.getContext('2d');
    const snap = () => g.getImageData(0,0,c.width,c.height).data.join(',');
    playerX = -14; playerZ = -14; drawMinimap(); const a = snap();
    playerX =  14; playerZ =  14; drawMinimap(); const b = snap();
    playerX = 0; playerZ = 0; drawMinimap();
    return a !== b;
  })()`);
  ck('내 위치가 미니맵에 반영된다', moved === true);

  // 1인칭 팔 — 어깨/팔꿈치/글러브가 이어져 있는가
  const armInfo = await f.evaluate(`(() => ({
    hasArms: typeof myArms === 'object' && !!myArms.L && !!myArms.R,
    hasPose: !!latestPose,
    // 포즈가 없으면 팔을 숨기는 것이 정상이다 (안 그러면 화면을 가로질러 늘어난다).
    // 규칙 자체를 확인한다 — 헤드리스에는 진짜 포즈가 없다.
    hiddenWithoutPose: !latestPose && !myArms.L.upper.visible && !myArms.R.upper.visible,
    // 포즈가 있다고 가정하고 한 프레임 돌리면 다시 보여야 한다
    shownWithPose: (() => {
      const save = latestPose;
      latestPose = new Array(33).fill({ x: 0.5, y: 0.5, z: 0, visibility: 1 });
      animateFrame(performance.now());
      const v = myArms.L.upper.visible && myArms.R.fore.visible && myLeftGlove.visible;
      latestPose = save;
      return v;
    })(),
    inRig: myArms.L.upper.parent === cameraRig && myArms.R.fore.parent === cameraRig,
    // 팔꿈치를 크게 움직이면 상완 길이가 따라 바뀌어야 한다 (실제로 이어져 있다는 증거)
    lenTest: (() => {
      myArms.R.elbow.position.set(2.0, -1.0, -2.0); layBone(myArms.R.upper, myArms.R.shoulder, myArms.R.elbow.position);
      const a = myArms.R.upper.scale.y;
      myArms.R.elbow.position.set(6.0, -4.0, -6.0); layBone(myArms.R.upper, myArms.R.shoulder, myArms.R.elbow.position);
      const b = myArms.R.upper.scale.y;
      return b > a * 1.5;
    })()
  }))()`);
  ck('1인칭 팔 관절이 만들어졌다', armInfo.hasArms);
  ck('포즈가 없으면 팔을 숨긴다 (화면을 가로질러 늘어나지 않게)',
     armInfo.hasPose ? true : armInfo.hiddenWithoutPose,
     armInfo.hasPose ? '포즈 있음' : '숨김 확인');
  ck('포즈가 들어오면 팔이 다시 보인다', armInfo.shownWithPose === true);
  ck('카메라 리그에 붙어 있다', armInfo.inRig);
  ck('팔꿈치를 움직이면 뼈 길이가 따라온다', armInfo.lenTest);

  // 가드 실드
  const sh = await f.evaluate(`(() => ({
    exists: typeof myShield === 'object' && !!myShield.material,
    hiddenAtStart: myShield.material.opacity < 0.02,
    facesForward: myShield.position.z < 0
  }))()`);
  ck('가드 실드가 존재한다', sh.exists);
  ck('평상시에는 안 보인다', sh.hiddenAtStart);
  ck('정면(-z)에 배치된다', sh.facesForward);

  console.log('');
  console.log('--- K.O. 가 화면에 드러나는가 (이동 불가 원인이 보여야 한다) ---');
  const koState = await f.evaluate(`(() => {
    const F = (hp) => ({ name:'x', hp, world_x:0, world_z:0, yaw:0 });
    socket.onmessage({ data: JSON.stringify({ fighters: {
      client_1: F(100), client_2: F(0), client_3: F(100), client_4: F(100) } }) });
    return {
      overlay: document.getElementById('ko-overlay').style.display,
      hpNum: document.getElementById('hp-num').innerText,
      hpWidth: document.getElementById('hp-fill').style.width,
      myHp: myHp
    };
  })()`);
  ck('HP 0 이면 K.O. 오버레이가 뜬다', koState.overlay === 'flex', koState.overlay);
  ck('HP 바가 0을 가리킨다', koState.hpNum === '0' && koState.hpWidth === '0%',
     `${koState.hpNum} / ${koState.hpWidth}`);
  const revived = await f.evaluate(`(() => {
    const F = (hp) => ({ name:'x', hp, world_x:0, world_z:0, yaw:0 });
    socket.onmessage({ data: JSON.stringify({ fighters: {
      client_1: F(100), client_2: F(100), client_3: F(100), client_4: F(100) } }) });
    return { overlay: document.getElementById('ko-overlay').style.display,
             hpNum: document.getElementById('hp-num').innerText };
  })()`);
  ck('부활하면 오버레이가 사라진다', revived.overlay === 'none', revived.overlay);
  ck('HP 바도 복구된다', revived.hpNum === '100', revived.hpNum);

  f.close();
  await sleep(700);

  console.log('');
  console.log('--- Host 아레나 페이지 ---');
  const a = await open(`${BASE}/arena`, { port: 9342, settle: 7000, fakeMedia: false });
  const ast = await a.evaluate(`(() => ({
    canvas: !!document.querySelector('#arena-canvas canvas'),
    fps: parseInt(document.getElementById('stat-fps').innerText) || 0,
    clock: document.getElementById('round-clock').innerText,
    modal: document.getElementById('result-modal').style.display
  }))()`);
  const aerrs = a.logs.filter(l => l.kind === 'EXCEPTION');

  ck('렌더 캔버스가 만들어진다', ast.canvas);
  ck('rAF 루프가 실제로 돈다', ast.fps > 0, `${ast.fps} FPS`);
  ck('런타임 예외 0건', aerrs.length === 0, aerrs.map(e => e.text).join(' / ') || 'none');
  ck('시계가 mm:ss 경과 시간 형식', /^\d+:\d\d$/.test(ast.clock), ast.clock);
  ck('시작하자마자 결과창이 뜨지 않는다', ast.modal !== 'flex');
  a.close();

  console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('하니스 오류:', e && e.message); process.exit(1); });
