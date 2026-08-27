/**
 * punch_harness.js — 펀치 판정 검증
 *
 *   cd iter3/tests && node punch_harness.js
 *
 * **런타임과 같은 파일을 그대로 import 한다** (`server/static/punch_core.js`).
 * 예전에는 fighter_client.html 에서 함수 원문을 텍스트로 오려내 돌렸는데,
 * 중괄호 매칭에 의존해 취약했고 무엇보다 "로직이 HTML 안에 있다"는 사실 자체가 문제였다.
 * 이제 브라우저와 Node 가 같은 모듈을 쓰므로 판정이 어긋날 여지가 없다.
 *
 * 검증 목표는 서로 반대 방향의 두 가지다.
 *   (A) 잽처럼 짧고 빠른 펀치가 **인식되는가**
 *   (B) 펀치가 아닌 동작이 **인식되지 않는가**
 * 임계값을 감으로 만지면 한쪽을 고치다 다른 쪽을 부순다. 이 하니스가 그 균형을 잡는다.
 */
const PunchCore = require('../server/static/punch_core.js');

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

const TUNE = PunchCore.PUNCH_TUNE;

/**
 * 팔 궤적을 만들어 흘려보낸다. 어깨는 고정, 손목이 뻗었다 돌아온다.
 * punchCore.kinematics 가 실제로 하는 계산(속도·뻗음 변화율)을 그대로 타도록
 * **좌표만** 넣는다 — 운동학 값을 직접 만들어 넣으면 그 계산은 검증되지 않는다.
 *
 * @param opts.side        'L' | 'R'
 * @param opts.startReach  가드 자세의 어깨→손목 거리 (어깨폭 배수)
 * @param opts.peakReach   최대로 뻗었을 때
 * @param opts.riseMs      뻗는 데 걸리는 시간 (잽은 짧다)
 * @param opts.sweepDeg    뻗는 동안 손목이 도는 각도. 직선 펀치는 0.
 *                         훅·어퍼는 손목이 **호를 그리며** 돌기 때문에, 뻗음 변화율보다
 *                         실제 손목 속도가 훨씬 크다. 직선으로 모델링하면 속도가 과소평가돼
 *                         발동 자체가 안 된다 (그렇게 만들었다가 실제로 안 걸렸다).
 * @param opts.sweepPlane  'h' 수평(훅) | 'v' 수직(어퍼)
 * @param opts.tilt        기본 방향의 미세 기울기 [x, y] (잽이 살짝 비스듬한 경우)
 * @param opts.elbowBend0/1  팔꿈치 굽힘 0(곧게 폄)~1(깊게 접음), 시작→끝.
 *                         분류는 "최고 속도 순간"의 각도를 쓰므로 이 차이가 종류를 가른다.
 */
function throwPunch(core, opts) {
  const side = opts.side || 'L';
  const dt = 1 / 30;                       // 포즈 추정 30fps
  const SH = 0.40;                         // 어깨폭(m)
  const rise = (opts.riseMs || 150) / 1000;
  const total = rise * 2.2;
  const t0 = opts.t0 || 10000;
  const shoulder = { x: side === 'L' ? -SH / 2 : SH / 2, y: 0, z: 0 };
  const sign = side === 'L' ? -1 : 1;      // 몸 바깥 방향
  const sweep = (opts.sweepDeg || 0) * Math.PI / 180;
  const tilt = opts.tilt || [0, 0];

  let result = null;
  for (let t = 0; t <= total; t += dt) {
    const u = Math.min(1, t / rise);
    // 뻗음 포락선은 sin^2 — 시작 순간의 미분이 0이라 최고 속도가 궤적 **중간**에서 나온다.
    // sin 을 쓰면 t=0 에서 미분이 최대라 "아직 옆을 향한 시점"이 최고 속도로 잡혀
    // 분류가 엉뚱해진다.
    const env = Math.pow(Math.sin(Math.min(t / total, 1) * Math.PI), 2);
    const reach = (opts.startReach + (opts.peakReach - opts.startReach) * env) * SH;

    // 기본 방향은 정면(-z). sweep 가 있으면 그 각도만큼 호를 그린다.
    //
    // **중앙을 가로지르게** 해야 한다 (+sweep/2 → -sweep/2).
    // 실제 훅은 바깥에서 들어와 상대를 지나쳐 가므로, 정면을 통과하는 임팩트 순간에
    // 속도가 순수 횡방향이 된다. 옆에서 정면으로 "모이기만" 하는 궤적으로 만들면
    // 그 순간 속도가 오히려 전방이라 STRAIGHT 로 분류된다 (그렇게 만들었다가 실제로 틀렸다).
    const a = (0.5 - u) * sweep;
    let ux, uy, uz;
    if (opts.sweepPlane === 'v') {
      // 수직 호 — 아래에서 위로 (어퍼컷). y는 아래가 +이므로 시작이 +y.
      ux = tilt[0];
      uy = Math.sin(a);
      uz = -Math.cos(a);
    } else {
      // 수평 호 — 바깥에서 안쪽으로 (훅)
      ux = sign * Math.sin(a) + tilt[0];
      uy = tilt[1];
      uz = -Math.cos(a);
    }
    const ul = Math.hypot(ux, uy, uz) || 1;
    const wrist = {
      x: shoulder.x + (ux / ul) * reach,
      y: shoulder.y + (uy / ul) * reach,
      z: shoulder.z + (uz / ul) * reach,
    };

    // 팔꿈치는 어깨-손목 중점에서 굽힘만큼 옆으로 빠진다 → 내각이 작아진다
    const b0 = opts.elbowBend0 || 0;
    const b1 = (opts.elbowBend1 !== undefined) ? opts.elbowBend1 : b0;
    const bend = b0 + (b1 - b0) * u;
    const elbow = {
      x: (shoulder.x + wrist.x) / 2 + sign * bend * SH * 0.9,
      y: (shoulder.y + wrist.y) / 2 + bend * SH * 0.35,
      z: (shoulder.z + wrist.z) / 2,
    };

    const now = t0 + t * 1000;
    const k = core.kinematics(side, shoulder, elbow, wrist, SH, now);
    const r = core.tryPunch(k, now);
    if (r) { result = r; break; }
  }
  return result;
}

const GUARD = 0.42;      // 가드 자세의 어깨→손목 거리 (어깨폭 배수)
const JAB   = { startReach: GUARD, peakReach: 1.05, riseMs: 130, tilt: [0.10, -0.10],
                elbowBend0: 0.55, elbowBend1: 0.05 };

console.log(`punch_core.js 를 직접 import — PUNCH_SPEED=${TUNE.PUNCH_SPEED} `
          + `REACH_N=${TUNE.PUNCH_REACH_N} WINDOW=${TUNE.PUNCH_WINDOW}\n`);

console.log('--- 단일 소스 확인 ---');
{
  const html = require('fs').readFileSync(
    require('path').join(__dirname, '../server/templates/fighter_client.html'), 'utf8');
  ck('fighter_client.html 에 판정 로직이 없다',
     !/function\s+(tryPunch|classifyPunch|armKinematics)\s*\(/.test(html));
  ck('fighter_client.html 에 펀치 임계값이 없다', !/PUNCH_SPEED\s*:/.test(html));
  ck('fighter_client.html 이 punch_core.js 를 로드한다', /punch_core\.js/.test(html));
  ck('런타임이 punchCore 를 통해 판정한다', /punchCore\.tryPunch\(/.test(html));
}

console.log('\n--- 잽: 짧고 빠른 정면 펀치 ---');
{
  let c = PunchCore.createPunchCore();
  const jab = throwPunch(c, Object.assign({ side: 'L' }, JAB));
  ck('빠른 잽이 인식된다', !!jab, jab ? jab.action : '미인식');
  ck('잽으로 분류된다 (훅/어퍼 아님)', jab && jab.kind === 'STRAIGHT', jab ? jab.kind : '-');

  c = PunchCore.createPunchCore();
  const short = throwPunch(c, { side: 'L', startReach: GUARD, peakReach: 0.92, riseMs: 110,
                                tilt: [0.12, -0.08], elbowBend0: 0.55, elbowBend1: 0.08 });
  ck('짧은 잽도 인식된다', !!short, short ? short.action : '미인식');

  c = PunchCore.createPunchCore();
  const cross = throwPunch(c, { side: 'R', startReach: GUARD, peakReach: 1.15, riseMs: 150,
                                tilt: [-0.08, -0.05], elbowBend0: 0.58, elbowBend1: 0.03 });
  ck('오른손 스트레이트도 인식', cross && cross.action === 'RIGHT_CROSS',
     cross ? cross.action : '미인식');
}

console.log('\n--- 훅 / 어퍼컷이 잽으로 오분류되지 않는가 ---');
{
  // 훅·어퍼는 팔을 **이미 접은 채** 휘두른다. 판정이 "최고 속도 순간"의 각도를 쓰므로
  // 이 차이가 분류를 가른다.
  let c = PunchCore.createPunchCore();
  const hook = throwPunch(c, { side: 'R', startReach: GUARD, peakReach: 0.95, riseMs: 190,
                               sweepDeg: 95, sweepPlane: 'h', elbowBend0: 0.62, elbowBend1: 0.55 });
  ck('훅이 인식된다', !!hook, hook ? hook.action : '미인식');
  ck('훅으로 분류된다', hook && hook.kind === 'HOOK', hook ? hook.kind : '-');

  c = PunchCore.createPunchCore();
  const upper = throwPunch(c, { side: 'R', startReach: GUARD, peakReach: 0.92, riseMs: 190,
                                sweepDeg: 85, sweepPlane: 'v', elbowBend0: 0.70, elbowBend1: 0.62 });
  ck('어퍼컷이 인식된다', !!upper, upper ? upper.action : '미인식');
  ck('어퍼컷으로 분류된다', upper && upper.kind === 'UPPERCUT', upper ? upper.kind : '-');
}

console.log('\n--- 오검출: 펀치가 아닌 동작 ---');
{
  let c = PunchCore.createPunchCore();
  const slow = throwPunch(c, { side: 'L', startReach: GUARD, peakReach: 0.95, riseMs: 900,
                               tilt: [0.1, 0.2], elbowBend0: 0.5, elbowBend1: 0.2 });
  ck('느린 팔 이동은 펀치가 아니다', !slow, slow ? slow.action : '미발동');

  c = PunchCore.createPunchCore();
  const twitch = throwPunch(c, { side: 'L', startReach: GUARD, peakReach: 0.58, riseMs: 110,
                                 tilt: [0.4, 0.3], elbowBend0: 0.6, elbowBend1: 0.55 });
  ck('뻗지 않는 잔동작은 펀치가 아니다', !twitch, twitch ? twitch.action : '미발동');

  c = PunchCore.createPunchCore();
  const still = throwPunch(c, { side: 'L', startReach: GUARD, peakReach: GUARD, riseMs: 150,
                                elbowBend0: 0.5, elbowBend1: 0.5 });
  ck('가만히 있으면 발동하지 않는다', !still);
}

console.log('\n--- 쿨다운 / 콤비네이션 ---');
{
  const c = PunchCore.createPunchCore();
  ck('첫 잽 발동', !!throwPunch(c, Object.assign({ side: 'L', t0: 10000 }, JAB)));
  ck('같은 팔 즉시 연타는 쿨다운에 막힌다',
     !throwPunch(c, Object.assign({ side: 'L', t0: 10150 }, JAB)));
  ck('쿨다운 후에는 다시 발동',
     !!throwPunch(c, Object.assign({ side: 'L', t0: 10800 }, JAB)));

  const c2 = PunchCore.createPunchCore();
  const j = throwPunch(c2, Object.assign({ side: 'L', t0: 20000 }, JAB));
  const cr = throwPunch(c2, { side: 'R', t0: 20330, startReach: GUARD, peakReach: 1.15,
                              riseMs: 150, tilt: [-0.08, -0.05],
                              elbowBend0: 0.58, elbowBend1: 0.03 });
  ck('원투 콤비네이션 (잽 → 스트레이트)', !!j && !!cr,
     `${j ? j.action : 'x'} → ${cr ? cr.action : 'x'}`);
}

console.log('\n--- 펀치 잠금 (isLocked) ---');
{
  const c = PunchCore.createPunchCore();
  ck('평상시에는 잠기지 않는다', c.isLocked(50000) === false);
  throwPunch(c, Object.assign({ side: 'L', t0: 50000 }, JAB));
  const fired = c.getLastPunchAny();
  ck('펀치 직후에는 잠긴다', c.isLocked(fired + 50) === true);
  ck(`최소 잠금(${TUNE.PUNCH_LOCK}ms) 직전까지 유지`, c.isLocked(fired + TUNE.PUNCH_LOCK - 10) === true);
  // 팔이 멈춘 상태로 만들면 최소 시간 이후 즉시 풀린다
  c.arms.L.lastSpeed = 0; c.arms.R.lastSpeed = 0;
  c.arms.L.lastReachN = 0.4; c.arms.R.lastReachN = 0.4;
  ck('팔이 회수되면 최소 시간 이후 풀린다', c.isLocked(fired + TUNE.PUNCH_LOCK + 10) === false);
  // 팔이 아직 나가 있으면 연장된다
  c.arms.L.lastSpeed = 1.5;
  ck('팔이 아직 나가 있으면 연장된다', c.isLocked(fired + TUNE.PUNCH_LOCK + 10) === true);
  ck('연장 상한을 넘으면 풀린다', c.isLocked(fired + TUNE.PUNCH_LOCK_MAX + 10) === false);
}

console.log('\n--- 임계값 주입 (튜닝·실험용) ---');
{
  // 같은 궤적이라도 임계값을 올리면 걸리지 않아야 한다 — 상수가 실제로 판정을 지배하는지 확인
  const strict = PunchCore.createPunchCore({ PUNCH_SPEED: 99 });
  ck('PUNCH_SPEED 를 올리면 발동하지 않는다',
     !throwPunch(strict, Object.assign({ side: 'L' }, JAB)));
  const loose = PunchCore.createPunchCore({ PUNCH_SPEED: 0.1, PUNCH_REACH_N: 0.1 });
  ck('내리면 발동한다', !!throwPunch(loose, Object.assign({ side: 'L' }, JAB)));
  ck('기본값은 오염되지 않는다', PunchCore.PUNCH_TUNE.PUNCH_SPEED === 1.6);
}

console.log('');
console.log('--- 필살기 (ENERGY_WAVE): 양손을 머리 위로 ---');
{
  /**
   * 필살기 자세를 유지한다. 속도를 요구하지 않는 **정적 자세**라 좌표만 세워 두면 된다.
   * @param opts.upN     손목이 코보다 얼마나 위인가 (어깨폭 배수). 음수면 손이 코 아래.
   * @param opts.reachN  팔이 얼마나 펴졌는가 (어깨폭 배수)
   * @param opts.oneHand true 면 왼손만 든다 (한 손으로는 안 걸려야 한다)
   * @param opts.holdMs  자세 유지 시간
   */
  function holdUlt(core, opts) {
    const SH = 0.40, dt = 1 / 30;
    const t0 = opts.t0 || 30000;
    const hold = opts.holdMs !== undefined ? opts.holdMs : 700;
    const canUse = opts.canUse !== undefined ? opts.canUse : true;
    const reach = (opts.reachN !== undefined ? opts.reachN : 0.95) * SH;
    const upL = (opts.upN !== undefined ? opts.upN : 0.55) * SH;
    const upR = opts.oneHand ? -0.30 * SH : upL;
    let out = null;

    for (let ms = 0; ms <= hold; ms += dt * 1000) {
      const now = t0 + ms;
      const mk = (side, up) => {
        const sh = { x: side === 'L' ? -SH / 2 : SH / 2, y: 0, z: 0 };
        // 코는 어깨선보다 조금 위 (world 는 y 가 아래로 +)
        const noseY = -0.55 * SH;
        const wr = { x: sh.x * 1.15, y: noseY - up, z: -0.12 * SH };
        // 팔 길이를 reach 에 맞춘다
        const d = Math.hypot(wr.x - sh.x, wr.y - sh.y, wr.z - sh.z) || 1;
        wr.x = sh.x + (wr.x - sh.x) / d * reach;
        wr.y = sh.y + (wr.y - sh.y) / d * reach;
        wr.z = sh.z + (wr.z - sh.z) / d * reach;
        const el = { x: (sh.x + wr.x) / 2, y: (sh.y + wr.y) / 2, z: (sh.z + wr.z) / 2 };
        return { k: core.kinematics(side, sh, el, wr, SH, now), noseY };
      };
      const L = mk('L', upL), R = mk('R', upR);
      const lUp = (L.noseY - (L.noseY - upL)) / SH;      // = upL / SH
      const rUp = opts.oneHand ? -0.30 : upL / SH;
      const r = core.tryUltimate(L.k, R.k, upL / SH, rUp, now, canUse);
      if (r) { out = { r, ms }; break; }
    }
    return out;
  }

  let c = PunchCore.createPunchCore();
  const ult = holdUlt(c, {});
  ck('양손을 머리 위로 들면 필살기가 나간다', !!ult, ult ? ult.r.action : '미발동');
  ck('액션 이름이 ENERGY_WAVE', ult && ult.r.action === PunchCore.ULTIMATE);
  ck('유지 시간 만큼 기다린 뒤 나간다', ult && ult.ms >= TUNE.ULT_HOLD_MS - 40,
     ult ? `${ult.ms | 0}ms (기준 ${TUNE.ULT_HOLD_MS})` : '-');

  c = PunchCore.createPunchCore();
  ck('게이지가 안 찼으면 발동하지 않는다', !holdUlt(c, { canUse: false }));

  c = PunchCore.createPunchCore();
  ck('한 손만 들면 발동하지 않는다', !holdUlt(c, { oneHand: true }));

  c = PunchCore.createPunchCore();
  ck('손이 코 아래면(가드 자세) 발동하지 않는다', !holdUlt(c, { upN: -0.20 }));

  c = PunchCore.createPunchCore();
  ck('머리를 감싸면(팔이 접힘) 발동하지 않는다', !holdUlt(c, { reachN: 0.45 }));

  c = PunchCore.createPunchCore();
  ck('잠깐 들었다 내리면 발동하지 않는다', !holdUlt(c, { holdMs: 200 }));

  // 충전 진행도
  c = PunchCore.createPunchCore();
  ck('자세를 잡기 전 충전은 0', c.ultCharge(1000) === 0);
  holdUlt(c, { t0: 5000, holdMs: 150 });
  ck('자세를 잡으면 충전이 올라간다', c.ultCharge(5150) > 0.2 && c.ultCharge(5150) < 1,
     c.ultCharge(5150).toFixed(2));

  // 쿨다운
  c = PunchCore.createPunchCore();
  ck('첫 필살기 발동', !!holdUlt(c, { t0: 40000 }));
  ck('쿨다운 안에는 다시 안 나간다', !holdUlt(c, { t0: 40500, holdMs: 600 }));
  ck('쿨다운 후에는 다시 나간다', !!holdUlt(c, { t0: 43000 }));

  // 펀치와 겹치지 않는가
  c = PunchCore.createPunchCore();
  const u2 = holdUlt(c, { t0: 60000 });
  ck('필살기 직후에는 펀치가 겹쳐 나가지 않는다',
     !!u2 && c.arms.L.lastPunch === c.getLastPunchAny());
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
