/**
 * avatar_page_harness.js — 아바타 비율 · 이동 보정 · 음향을 실제 페이지에서 확인한다.
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd iter3/tests && node avatar_page_harness.js [베이스URL]
 *
 * 로직 하니스로는 못 보는 것들을 본다.
 *   - 아바타가 사람 비율인가 (머리/키 비율, 다리 길이 비중)
 *   - 기본 자세가 복싱 가드인가 (팔을 늘어뜨리고 있으면 복서로 안 보인다)
 *   - 이동 보정 마법사가 임계값을 실제로 바꾸는가
 *   - 음향 모듈이 붙어 있고 데미지에 반응하는가
 */
const { open, sleep } = require('./_cdp');
const BASE = process.argv[2] || 'https://localhost:8000';

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

const HEAD_R = Number(
  require('fs').readFileSync(require('path').join(__dirname, '../server/static/humanoid.js'), 'utf8')
    .match(/const HEAD_R = ([\d.]+)/)[1]);

(async () => {
  console.log(`--- 아바타 비율 (Host 화면) · humanoid.js HEAD_R=${HEAD_R} ---`);
  const a = await open(`${BASE}/arena`, { port: 9381, settle: 7000, fakeMedia: false });

  const geo = await a.evaluate(`(() => {
    const h = fighterMeshes['client_1'];
    const box = new THREE.Box3().setFromObject(h.group);
    const height = box.max.y - box.min.y;

    // 어깨 관절 좌우 간격 = 어깨폭
    const shoulderSpan = Math.abs(h.armR.shoulder.position.x - h.armL.shoulder.position.x);
    const hipY = h.legL.hip.position.y;
    return {
      height,
      headY: h.head.position.y,
      headScale: [h.head.scale.x, h.head.scale.y, h.head.scale.z],
      shoulderSpan,
      hipY,
      footBottom: box.min.y,
      legRatio: (hipY - box.min.y) / height,   // 고관절에서 발 밑면까지
      // 기본 자세의 팔 각도 (음수 = 손을 올린 상태)
      armSX: h.armR.shoulder.rotation.x,
      elbowEX: h.armR.elbow.rotation.x,
      kneeX: h.legL.knee.rotation.x,
      // 스탠스: 앞뒤 발이 어긋나 있는가
      footStagger: Math.abs(h.legL.hip.position.z - h.legR.hip.position.z),
    };
  })()`);

  ck('아바타 키가 게임 스케일에 맞는다 (6~8)', geo.height > 6 && geo.height < 8.4,
     geo.height.toFixed(2));
  ck('다리가 키의 40% 이상 (예전 24%)', geo.legRatio >= 0.40,
     `${(geo.legRatio * 100).toFixed(0)}%`);
  const headDia = HEAD_R * geo.headScale[0] * 2;
  ck('머리가 어깨폭보다 작다 (예전엔 어깨폭만 했다)', headDia < geo.shoulderSpan,
     `머리지름 ${headDia.toFixed(2)} vs 어깨폭 ${geo.shoulderSpan.toFixed(2)}`);
  ck('머리가 키의 1/4 이하 (예전 1/2.7)', headDia / geo.height < 0.25,
     `1/${(geo.height / headDia).toFixed(1)}`);

  console.log('');
  console.log('--- 복싱 스탠스 ---');
  ck('기본 자세에서 손을 올리고 있다 (어깨 회전 < -0.7)', geo.armSX < -0.7, geo.armSX.toFixed(2));
  ck('팔꿈치가 접혀 있다 (< -1.8)', geo.elbowEX < -1.8, geo.elbowEX.toFixed(2));
  ck('무릎이 굽어 있다 (> 0.2)', geo.kneeX > 0.2, geo.kneeX.toFixed(2));
  ck('앞뒤 발이 어긋나 있다 (bladed stance)', geo.footStagger > 0.4, geo.footStagger.toFixed(2));

  console.log('');
  console.log('--- 음향 (Host) ---');
  const snd = await a.evaluate(`(() => ({
    lib: typeof window.createSound,
    inst: typeof snd,
    hasImpact: typeof (snd || {}).impact,
    hasKo: typeof (snd || {}).ko,
    noThrow: (() => { try { snd.impact(8, false); snd.whoosh(30); snd.ko(); snd.bell(); return true; }
                      catch (e) { return String(e); } })()
  }))()`);
  ck('sound.js 가 로드된다', snd.lib === 'function');
  ck('host 가 사운드 인스턴스를 갖는다', snd.inst === 'object' && snd.hasImpact === 'function');
  ck('소리 호출이 예외를 던지지 않는다 (무음 환경 포함)', snd.noThrow === true, String(snd.noThrow));

  const aerrs = a.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', aerrs.length === 0, aerrs.map(e => e.text).join(' | ') || 'none');
  a.close();
  await sleep(700);

  console.log('');
  console.log('--- 이동 범위 보정 (Fighter) ---');
  const f = await open(`${BASE}/client?id=client_3`, { port: 9382, settle: 9000 });

  const cal = await f.evaluate(`(() => {
    const before = { on: TUNE.PITCH_ON, back: TUNE.PITCH_BACK_ON, ready: moveCal.ready };
    // 마법사를 거치지 않고 측정값만 주입해 적용 경로를 확인한다
    moveCal.fwd = 0.40; moveCal.back = 0.24; moveCal.ready = true;
    applyMoveCal();
    const after = { on: TUNE.PITCH_ON, back: TUNE.PITCH_BACK_ON,
                    off: TUNE.PITCH_OFF, backOff: TUNE.PITCH_BACK_OFF };
    return { before, after,
      hasWizard: typeof calOpen === 'function' && typeof calFrame === 'function',
      btn: !!document.getElementById('cal-btn'),
      box: !!document.getElementById('cal-cap') };
  })()`);

  ck('보정 UI 가 있다', cal.btn && cal.box);
  ck('보정 함수가 있다', cal.hasWizard);
  ck('측정값이 임계값에 반영된다', Math.abs(cal.after.on - 0.40 * 0.45) < 1e-6,
     `전진 ${cal.before.on} → ${cal.after.on.toFixed(3)} (0.40 × 0.45)`);
  ck('후진도 내 범위로 잡힌다', Math.abs(cal.after.back - 0.24 * 0.45) < 1e-6,
     `후진 ${cal.after.back.toFixed(3)} (0.24 × 0.45)`);
  ck('해제 임계는 발동보다 낮다 (히스테리시스)',
     cal.after.off < cal.after.on && cal.after.backOff < cal.after.back,
     `${cal.after.off.toFixed(3)} < ${cal.after.on.toFixed(3)}`);
  ck('사람마다 다른 값이 나온다 (고정값이 아니다)',
     cal.after.on !== cal.before.on || cal.before.ready,
     `기본 ${cal.before.on} → 내 범위 ${cal.after.on.toFixed(3)}`);

  console.log('');
  console.log('--- 음향 (Fighter) ---');
  const fsnd = await f.evaluate(`(() => ({
    inst: typeof snd,
    muteBtn: !!document.getElementById('mute-btn'),
    noThrow: (() => { try { snd.impact(6, true); snd.hurt(6); snd.shield(); snd.step(); return true; }
                      catch (e) { return String(e); } })(),
    muteWorks: (() => { const m0 = snd.isMuted(); snd.setMuted(true); const m1 = snd.isMuted();
                        snd.setMuted(m0); return m1 === true; })()
  }))()`);
  ck('1인칭도 같은 사운드 모듈을 쓴다', fsnd.inst === 'object');
  ck('음소거 버튼이 있다', fsnd.muteBtn);
  ck('가드/피격/스텝 소리가 예외 없이 난다', fsnd.noThrow === true, String(fsnd.noThrow));
  ck('음소거 토글이 동작한다', fsnd.muteWorks === true);

  console.log('');
  console.log('--- 분노 게이지 · 불꽃 오라 · 필살기 (Fighter) ---');
  const rage = await f.evaluate(`(() => {
    const before = { num: document.getElementById('rage-num').innerText,
                     full: document.getElementById('rage-track').classList.contains('full') };
    renderRage(45);
    const mid = { w: document.getElementById('rage-fill').style.width,
                  num: document.getElementById('rage-num').innerText,
                  full: document.getElementById('rage-track').classList.contains('full') };
    renderRage(100);
    const top = { w: document.getElementById('rage-fill').style.width,
                  full: document.getElementById('rage-track').classList.contains('full'),
                  label: document.getElementById('rage-label').innerText };
    // 상대 아바타 오라
    const opp = opponentMeshes[Object.keys(opponentMeshes)[0]];
    opp.setRage(0); opp.update();
    const auraOff = opp.getRage();
    opp.setRage(100);
    for (let i = 0; i < 90; i++) opp.update();
    return { before, mid, top, auraOff, auraOn: opp.getRage(),
             hasSetRage: typeof opp.setRage === 'function',
             ultConst: window.PunchCore.ULTIMATE,
             hasTryUlt: typeof punchCore.tryUltimate };
  })()`);
  ck('게이지 45 → 폭 45%', rage.mid.w === '45%', rage.mid.w);
  ck('45 는 "가득 참"이 아니다', rage.mid.full === false);
  ck('게이지 100 → 폭 100% + 준비 표시', rage.top.w === '100%' && rage.top.full === true);
  ck('가득 차면 필살기 안내가 뜬다', /필살기 준비/.test(rage.top.label), rage.top.label.trim());
  ck('아바타에 오라 API 가 있다', rage.hasSetRage === true);
  ck('오라가 게이지를 따라간다', rage.auraOff === 0 && rage.auraOn === 1,
     `${rage.auraOff} → ${rage.auraOn}`);
  ck('필살기 판정이 코어에 있다', rage.hasTryUlt === 'function' && rage.ultConst === 'ENERGY_WAVE',
     rage.ultConst);

  const ferrs = f.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', ferrs.length === 0, ferrs.map(e => e.text).join(' | ') || 'none');
  f.close();

  console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('하니스 오류:', e && e.message); process.exit(1); });
