/** 휴리스틱 최적화 브라우저 엔진의 로드·시퀀스·팔/종류 재분류 회귀 테스트. */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.join(__dirname, '..');
const artifact = JSON.parse(fs.readFileSync(
  path.join(root, 'server/static/models/heuristic_thresholds.json'), 'utf8'
));
const source = fs.readFileSync(path.join(root, 'server/static/heuristic_engine.js'), 'utf8');
const context = {
  console,
  fetch: async () => ({ ok: true, status: 200, json: async () => artifact }),
};
context.window = context;
vm.createContext(context);
vm.runInContext(source, context);

function check(message, condition) {
  if (!condition) throw new Error(`FAIL: ${message}`);
  console.log(`PASS: ${message}`);
}

function row(overrides = {}) {
  const values = new Float32Array(17);
  values[0] = values[1] = 0.7;
  values[12] = 1.4;
  values[15] = 2.0;
  Object.entries(overrides).forEach(([index, value]) => { values[Number(index)] = value; });
  return values;
}

(async () => {
  check('배포 artifact가 heuristic_7j_v1을 사용한다', artifact.feature_set === 'heuristic_7j_v1');
  const engine = context.createOptimizedHeuristicEngine();
  await engine.load('/static/models/heuristic_thresholds.json');
  check('임계값 파일 로드 후 ready', engine.ready === true);

  // 양팔 속도가 낮고 가드 거리 조건도 아닌 시퀀스는 IDLE이어야 한다.
  for (let i = 0; i < 12; i++) engine.push(row(), i * 25);
  check('저속 시퀀스 IDLE 안정화', engine.stableState()?.label === 'IDLE');

  engine.reset();
  // 왼손 peak에서 위쪽 속도가 지배적이고 팔꿈치가 접힌 합성 시퀀스.
  for (let i = 0; i < 12; i++) {
    engine.push(row({ 0: 0.4, 5: -12, 10: 12, 11: 1 }), 500 + i * 25);
  }
  const guess = engine.guessPunchKind('L');
  check('왼손 어퍼컷 종류 재분류', guess?.action === 'LEFT_UPPERCUT');
  check('반대 팔 요청은 재분류하지 않음', engine.guessPunchKind('R') === null);

  console.log('heuristic_engine_harness: all checks passed');
})().catch((error) => { console.error(error); process.exit(1); });
