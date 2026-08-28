/**
 * hybrid_engine_harness.js — 'hybrid' 엔진 모드(v5c: TCN 확신도 AND 룰베이스 물리조건) 검증.
 *
 * fighter_client.html 에서 실제 판정 분기 원문을 통째로 추출해(로직을 베껴 쓰지 않음) 구동한다
 * (move_harness.js와 같은 방식). 목표:
 *   (A) ENGINE_REGISTRY에 'hybrid' 키가 등록되어 있고 tcnEngine을 공유하는지
 *   (B) 'hybrid' 모드에서 TCN이 확신하지 못하면(guessPunchKind() === null) 룰베이스가 이미
 *       물리조건을 통과시킨 punch라도 **무효화**되는지 (v5b/v5c 실험이 확인한 AND 게이트)
 *   (C) 'tcn' 모드는 여전히 같은 상황에서 룰베이스 종류로 **폴백**해 punch를 유지하는지
 *       (회귀 방지 — hybrid를 추가하다 기존 tcn 동작을 건드리지 않았는지)
 *   (D) 'hybrid' 모드에서 TCN이 확신하면 punch.action이 TCN 판단으로 바뀌는지
 *
 * cd iter4/tests && node hybrid_engine_harness.js
 */
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '../server/templates/fighter_client.html'), 'utf8');

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

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

/** extract()는 첫 { }블록에서 멈춘다 — if/else if/else 체인 전체를 원문 그대로 쓸어 담는다. */
function extractIfElseChain(startPattern) {
  const i = html.indexOf(startPattern);
  if (i < 0) throw new Error(`찾지 못함: ${startPattern}`);
  let depth = 0, started = false, j = i;
  for (; j < html.length; j++) {
    if (html[j] === '{') { depth++; started = true; }
    else if (html[j] === '}') {
      depth--;
      if (started && depth === 0) {
        j++;
        // 다음 비공백 토큰이 'else'면 그 블록까지 계속 이어붙인다.
        const rest = html.slice(j).match(/^\s*/)[0];
        if (html.slice(j + rest.length, j + rest.length + 4) === 'else') {
          started = false; // 다음 { 를 새 블록 시작으로 다시 잡는다
          continue;
        }
        break;
      }
    }
  }
  return html.slice(i, j);
}

// --- (A) ENGINE_REGISTRY 원문 그대로 실행 ---
const punchCore = { ULTIMATE: 'ENERGY_WAVE' };
const tcnEngine = { _mockGuess: null, guessPunchKind() { return this._mockGuess; } };
const createConsensusEngine = () => ({ guessPunchKind: () => null }); // 이 하니스에서는 안 씀
const heuristicNoAugEngine = {}, heuristicAugEngine = {};

const registrySrc = extract('const ENGINE_REGISTRY = {');
const ENGINE_REGISTRY = eval(`(${registrySrc.replace(/^const ENGINE_REGISTRY = /, '')})`);

ck("ENGINE_REGISTRY에 'hybrid' 키가 있다", 'hybrid' in ENGINE_REGISTRY);
ck("hybrid는 tcnEngine을 공유한다 (같은 모델, 같은 buffer)", ENGINE_REGISTRY.hybrid.engine === tcnEngine);
ck("tcn 모드도 여전히 있다 (회귀 방지)", 'tcn' in ENGINE_REGISTRY && ENGINE_REGISTRY.tcn.engine === tcnEngine);

// --- (B)(C)(D) 판정 분기 원문 실행 ---
// 원문을 그대로 함수로 감싼다 — 로직을 베껴 쓰지 않고 실제 if/else 문자열을 그대로 돈다.
const guessBlockSrc = extractIfElseChain("if (punch && selectedEngine && punch.action !== punchCore.ULTIMATE) {");

function runGuessBlock({ motionMode, punch, mockGuess }) {
  const selectedEngine = tcnEngine;
  tcnEngine._mockGuess = mockGuess;
  let punchLabel = '', punchLabelT = 0;
  const now = 12345;
  const fn = new Function(
    'punch', 'selectedEngine', 'punchCore', 'motionMode', 'now',
    `let punchLabel, punchLabelT;
     ${guessBlockSrc}
     return { punch, punchLabel, punchLabelT };`
  );
  return fn(punch, selectedEngine, punchCore, motionMode, now);
}

// (B) hybrid 모드, TCN 미확신 -> 무효화
{
  const original = { action: 'LEFT_JAB', kind: 'STRAIGHT', speed: 2.0 };
  const res = runGuessBlock({ motionMode: 'hybrid', punch: Object.assign({}, original), mockGuess: null });
  ck('hybrid + TCN 미확신 -> punch가 null로 무효화된다', res.punch === null, JSON.stringify(res.punch));
  ck('hybrid + TCN 미확신 -> 거부 사유가 HUD 라벨에 남는다', /REJECTED/.test(res.punchLabel), res.punchLabel);
}

// (C) tcn 모드, TCN 미확신 -> 룰베이스 종류로 폴백(무효화하지 않음) — 회귀 방지
{
  const original = { action: 'LEFT_JAB', kind: 'STRAIGHT', speed: 2.0 };
  const res = runGuessBlock({ motionMode: 'tcn', punch: Object.assign({}, original), mockGuess: null });
  ck('tcn + TCN 미확신 -> punch가 유지된다(폴백, 무효화 아님)', res.punch !== null && res.punch.action === 'LEFT_JAB', JSON.stringify(res.punch));
}

// (D) hybrid 모드, TCN 확신 -> TCN 판단으로 교체되고 유지된다
{
  const original = { action: 'LEFT_JAB', kind: 'STRAIGHT', speed: 2.0 };
  const guess = { action: 'LEFT_HOOK', kind: 'HOOK', confidence: 0.91 };
  const res = runGuessBlock({ motionMode: 'hybrid', punch: Object.assign({}, original), mockGuess: guess });
  ck('hybrid + TCN 확신 -> punch가 TCN 종류로 갱신되고 유지된다',
     res.punch !== null && res.punch.action === 'LEFT_HOOK' && res.punch.kind === 'HOOK',
     JSON.stringify(res.punch));
}

// (E) 필살기는 hybrid에서도 절대 건드리지 않는다 (punch.action === punchCore.ULTIMATE 이면 분기 자체를 안 탐)
{
  const original = { action: 'ENERGY_WAVE', kind: 'ULTIMATE', speed: 1.0 };
  const res = runGuessBlock({ motionMode: 'hybrid', punch: Object.assign({}, original), mockGuess: null });
  ck('hybrid에서도 필살기는 TCN 게이트를 안 탄다', res.punch !== null && res.punch.action === 'ENERGY_WAVE', JSON.stringify(res.punch));
}

console.log(fail === 0 ? `\n✅ 전부 통과` : `\n❌ ${fail}건 실패`);
process.exit(fail === 0 ? 0 : 1);
