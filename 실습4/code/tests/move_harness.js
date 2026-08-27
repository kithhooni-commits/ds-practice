/**
 * move_harness.js — 펀치 잠금 중 이동 처리 검증
 *
 * fighter_client.html 에서 TUNE / vote / updateStates 의 **원문을 그대로 추출**해 구동한다.
 * 검증 목표는 서로 반대 방향의 두 가지다.
 *
 *   (A) 3·5차 회귀 방지 — 펀치가 나를 밀어서는 안 된다.
 *       가만히 선 채 연타하면, 펀치가 자세 신호를 아무리 흔들어도 FORWARD/BACK 이 생기지 않아야 한다.
 *   (B) 7차 수정     — 연타 중에도 이미 하고 있던 전진은 유지돼야 한다.
 *       펀치 간격(0.20~0.30s)이 잠금 하한보다 짧으면 예전 코드는 이동이 영영 0이었다.
 *
 *   cd iter3/tests && node move_harness.js
 */
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '../server/templates/fighter_client.html'), 'utf8');

/** 중괄호 균형으로 선언 원문을 통째로 잘라낸다 (테스트에 로직을 베껴 쓰지 않기 위해). */
function extract(startPattern) {
  const i = html.indexOf(startPattern);
  if (i < 0) throw new Error(`찾지 못함: ${startPattern}`);
  let depth = 0, started = false, j = i;
  for (; j < html.length; j++) {
    if (html[j] === '{') { depth++; started = true; }
    else if (html[j] === '}') { depth--; if (started && depth === 0) { j++; break; } }
  }
  return html.slice(i, j);
}

const src = `
  ${extract('const TUNE = {')};
  const clamp01 = v => (v < 0 ? 0 : (v > 1 ? 1 : v));
  let sRoll = 0, sPitch = 0, sShift = 0;
  let moveState = "NONE", rotState = "NONE";
  let moveIntensity = 0, rotIntensity = 0;
  let punchLocked = false, lastRotInputT = 0;
  let wasLocked = false;
  const lockedMove = { state: "NONE", intensity: 0 };
  const moveVote = { cand: "NONE", since: 0 }, rotVote = { cand: "NONE", since: 0 };
  ${extract('function vote(')}
  ${extract('function updateStates(')}
  return {
    TUNE,
    set: o => { if ('sRoll' in o) sRoll = o.sRoll; if ('sPitch' in o) sPitch = o.sPitch;
                if ('sShift' in o) sShift = o.sShift; if ('locked' in o) punchLocked = o.locked; },
    step: (now, dt) => updateStates(now, dt),
    get: () => ({ moveState, moveIntensity, rotState, rotIntensity }),
  };
`;
const sim = new Function(src)();
const TUNE = sim.TUNE;

let fail = 0;
const ck = (name, cond, extra) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${extra !== undefined ? `  (${extra})` : ''}`);
  if (!cond) fail++;
};

console.log(`설정: PUNCH_LOCK=${TUNE.PUNCH_LOCK}ms  PITCH_ON=${TUNE.PITCH_ON}  PITCH_BACK_ON=${TUNE.PITCH_BACK_ON}\n`);

const DT = 1 / 30;           // 포즈 추정 ~30fps
const PUNCH_EVERY = 0.25;    // 실측 중앙값 — 로그에서 확인한 연타 간격

/**
 * 연타를 그대로 재현한다. PUNCH_EVERY 마다 펀치가 나가고, 그때부터 PUNCH_LOCK 동안 잠긴다.
 * pitchDuring: 잠금 구간에서 자세 신호가 어떻게 오염되는지 (펀치는 몸통을 틀어 pitch를 흔든다)
 */
function run({ seconds, pitchIdle, pitchDuring, rollIdle = 0 }) {
  let now = 0, lastPunch = -999, advance = 0, retreat = 0;
  const seen = new Set();
  for (let i = 0; i < seconds / DT; i++) {
    now += DT * 1000;
    if (now - lastPunch >= PUNCH_EVERY * 1000) lastPunch = now;
    const locked = (now - lastPunch) < TUNE.PUNCH_LOCK;
    sim.set({ locked, sRoll: rollIdle, sPitch: locked ? pitchDuring : pitchIdle });
    sim.step(now, DT);
    const g = sim.get();
    seen.add(g.moveState);
    if (g.moveState === 'FORWARD') advance += g.moveIntensity * DT;
    if (g.moveState === 'BACK')    retreat += g.moveIntensity * DT;
  }
  return { advance, retreat, seen: [...seen], last: sim.get() };
}

console.log('--- (A) 가만히 선 채 연타 — 펀치가 나를 밀면 안 된다 ---');
{
  // 정지 상태(pitch 0)에서 펀치할 때마다 몸통이 틀려 pitch가 -0.30 까지 흔들리는 최악 가정
  const r = run({ seconds: 6, pitchIdle: 0.0, pitchDuring: -0.30 });
  ck('BACK 이 한 번도 안 생긴다', !r.seen.includes('BACK'), r.seen.join('/'));
  ck('뒤로 밀린 거리 0', r.retreat < 1e-9, r.retreat.toFixed(4));
  ck('앞으로도 안 밀린다', r.advance < 1e-9, r.advance.toFixed(4));
}

console.log('\n--- (A-2) 오염이 전진 방향일 때도 마찬가지 ---');
{
  const r = run({ seconds: 6, pitchIdle: 0.0, pitchDuring: +0.40 });
  ck('FORWARD 가 안 생긴다', !r.seen.includes('FORWARD'), r.seen.join('/'));
  ck('전진 거리 0', r.advance < 1e-9, r.advance.toFixed(4));
}

console.log('\n--- (B) 전진 중 연타 — 이동이 유지돼야 한다 ---');
{
  // 확실히 앞으로 숙인 자세(pitch 0.30 > PITCH_ON 0.16)를 유지한 채 0.25초마다 펀치
  const r = run({ seconds: 6, pitchIdle: 0.30, pitchDuring: -0.30 });
  ck('FORWARD 상태가 유지된다', r.seen.includes('FORWARD'), r.seen.join('/'));
  ck('실제로 앞으로 나간다', r.advance > 1.0, `누적 강도·시간 ${r.advance.toFixed(2)}`);
  ck('뒤로는 안 간다', r.retreat < 1e-9, r.retreat.toFixed(4));
  const units = r.advance * 14.0;   // MOVE_SPEED
  console.log(`       => 6초 연타 중 전진량 약 ${units.toFixed(1)} units (스폰 17u 거리 기준)`);
  ck('교전 사거리(17u -> 9u, 8u 이동)를 6초 안에 좁힐 수 있다', units >= 8, `${units.toFixed(1)}u`);
}

console.log('\n--- (C) 예전 동작(잠금 중 강도 0)이었다면 어땠는가 ---');
{
  // 잠금 중 mi=0 으로 죽이던 옛 로직을 그대로 재현해 비교한다
  let now = 0, lastPunch = -999, advance = 0, mIntensity = 0, mState = 'NONE';
  const vote2 = { cand: 'NONE', since: 0 };
  for (let i = 0; i < 6 / DT; i++) {
    now += DT * 1000;
    if (now - lastPunch >= PUNCH_EVERY * 1000) lastPunch = now;
    const locked = (now - lastPunch) < 480;       // 옛 PUNCH_LOCK
    const sPitch = locked ? -0.30 : 0.30;
    let cand = 'NONE';
    if (!locked) {
      const on = (mState === 'FORWARD') ? TUNE.PITCH_OFF : TUNE.PITCH_ON;
      if (sPitch > on) cand = 'FORWARD';
    }
    if (cand !== vote2.cand) { vote2.cand = cand; vote2.since = now; }
    const need = (cand === 'NONE') ? TUNE.HOLD_OFF_MS : TUNE.HOLD_ON_MS;
    mState = (now - vote2.since >= need) ? cand : mState;
    const mi = locked ? 0 : (mState === 'FORWARD'
      ? Math.min(1, Math.max(0, (sPitch - TUNE.PITCH_OFF) / TUNE.PITCH_RANGE)) : 0);
    mIntensity += (mi - mIntensity) * (1 - Math.exp(-DT / 0.13));
    if (mState === 'FORWARD') advance += mIntensity * DT;
  }
  const units = advance * 14.0;
  console.log(`       옛 로직 6초 전진량: ${units.toFixed(1)} units`);
  ck('옛 로직은 사실상 전진 불가였다 (< 2u)', units < 2, `${units.toFixed(1)}u`);
}

console.log('\n--- (D) 연타를 멈추면 즉시 정상 판정으로 복귀 ---');
{
  sim.set({ locked: false, sPitch: 0, sRoll: 0 });
  for (let i = 0, t = 100000; i < 60; i++) { t += DT * 1000; sim.step(t, DT); }
  ck('정지 자세면 NONE', sim.get().moveState === 'NONE', sim.get().moveState);
  let t = 200000;
  sim.set({ locked: false, sPitch: -0.30 });
  for (let i = 0; i < 60; i++) { t += DT * 1000; sim.step(t, DT); }
  ck('잠금이 풀린 뒤 진짜 후진은 정상 인식', sim.get().moveState === 'BACK', sim.get().moveState);
}

console.log('');
console.log('--- 전진/후진 비대칭 (실사용 피드백: 전진이 너무 빡빡했다) ---');
{
  /** 주어진 pitch 를 계속 유지하면 어떤 상태로 확정되는지 */
  let clock = 500000;
  function settle(pitch) {
    // 반드시 중립(NONE)에서 출발해야 한다. 상태 머신에 히스테리시스가 있어
    // 이전 상태가 남아 있으면 PITCH_ON 이 아니라 PITCH_OFF 로 판정돼 측정이 오염된다.
    sim.set({ locked: false, sRoll: 0, sPitch: 0 });
    for (let i = 0; i < 60; i++) { clock += DT * 1000; sim.step(clock, DT); }
    sim.set({ locked: false, sRoll: 0, sPitch: pitch });
    for (let i = 0; i < 90; i++) { clock += DT * 1000; sim.step(clock, DT); }
    return sim.get().moveState;
  }
  /** 상태가 켜지는 최소 |pitch| 를 이분 탐색으로 찾는다 */
  function threshold(sign, want) {
    let lo = 0, hi = 0.5;
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) / 2;
      if (settle(sign * mid) === want) hi = mid; else lo = mid;
    }
    return hi;
  }
  const fwd = threshold(1, 'FORWARD');
  const back = threshold(-1, 'BACK');
  console.log('       전진 발동 임계 ' + fwd.toFixed(3) + ' · 후진 발동 임계 ' + back.toFixed(3));
  ck('전진이 후진보다 쉽게 걸린다', fwd < back, fwd.toFixed(3) + ' < ' + back.toFixed(3));
  ck('전진 임계가 충분히 낮다 (<=0.12)', fwd <= 0.12, fwd.toFixed(3));
  ck('후진 임계가 충분히 높다 (>=0.15)', back >= 0.15, back.toFixed(3));
  ck('설정값과 일치 (전진)', Math.abs(fwd - TUNE.PITCH_ON) < 0.01,
     fwd.toFixed(3) + ' vs ' + TUNE.PITCH_ON);
  ck('설정값과 일치 (후진)', Math.abs(back - TUNE.PITCH_BACK_ON) < 0.01,
     back.toFixed(3) + ' vs ' + TUNE.PITCH_BACK_ON);
  ck('중립(0)에서는 아무 상태도 안 걸린다', settle(0) === 'NONE', settle(0));
}

console.log('');
console.log('--- 이동 반응 지연 (펀치는 빠른데 이동만 느리다는 피드백) ---');
{
  /**
   * 중립에서 시작해 pitch 를 걸었을 때 **실제로 움직이기 시작할 때까지** 걸리는 시간(ms).
   * 상태가 확정돼도 강도가 0 이면 안 움직이므로, 강도가 의미 있게 붙는 시점을 잰다.
   */
  function latency(pitch, want, minIntensity) {
    sim.set({ locked: false, sRoll: 0, sPitch: 0 });
    let t = 900000;
    for (let i = 0; i < 60; i++) { t += DT * 1000; sim.step(t, DT); }   // 중립 안정화
    sim.set({ locked: false, sRoll: 0, sPitch: pitch });
    const t0 = t;
    for (let i = 0; i < 200; i++) {
      t += DT * 1000;
      sim.step(t, DT);
      const g = sim.get();
      if (g.moveState === want && g.moveIntensity >= (minIntensity || 0.25)) return t - t0;
    }
    return Infinity;
  }

  // 살짝 넘긴 경우 (투표를 거쳐야 한다)
  const slow = latency(TUNE.PITCH_ON * 1.15, 'FORWARD');
  // 확실히 크게 숙인 경우 (즉시 확정 경로)
  const fast = latency(TUNE.PITCH_ON * 2.2, 'FORWARD');
  console.log(`       살짝 숙임 ${slow.toFixed(0)}ms · 크게 숙임 ${fast.toFixed(0)}ms`);

  ck('크게 숙이면 즉시 반응한다 (<=120ms)', fast <= 120, `${fast.toFixed(0)}ms`);
  ck('살짝 숙여도 예전보다 빠르다 (<=320ms)', slow <= 320, `${slow.toFixed(0)}ms`);
  ck('크게 숙인 쪽이 더 빠르다', fast < slow, `${fast.toFixed(0)} < ${slow.toFixed(0)}`);

  // 좌우도 같은 경로를 타는가
  function latencyRoll(roll, want) {
    sim.set({ locked: false, sRoll: 0, sPitch: 0 });
    let t = 950000;
    for (let i = 0; i < 60; i++) { t += DT * 1000; sim.step(t, DT); }
    sim.set({ locked: false, sRoll: roll, sPitch: 0 });
    const t0 = t;
    for (let i = 0; i < 200; i++) {
      t += DT * 1000; sim.step(t, DT);
      const g = sim.get();
      if (g.moveState === want && g.moveIntensity >= 0.25) return t - t0;
    }
    return Infinity;
  }
  const rollFast = latencyRoll(TUNE.ROLL_ON * 2.2, 'LEFT');
  ck('좌우 스텝도 즉시 반응한다 (<=120ms)', rollFast <= 120, `${rollFast.toFixed(0)}ms`);

  // 오검출 방어가 살아 있는가 — 임계값 바로 아래는 여전히 안 걸린다
  ck('임계값 아래는 여전히 발동하지 않는다',
     latency(TUNE.PITCH_ON * 0.85, 'FORWARD') === Infinity);
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
