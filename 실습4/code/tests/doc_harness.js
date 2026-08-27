/**
 * doc_harness.js — README 에 적힌 수치가 코드와 일치하는지 확인한다.
 *
 *   cd iter3/tests && node doc_harness.js
 *
 * 이 하니스가 있는 이유: README 가 반복적으로 코드와 어긋났다.
 * 데미지 표는 밸런스를 바꾼 뒤에도 옛 값(12/16/18/25)이 남아 있었고,
 * pitch 임계값은 뒤집은 뒤에도 옛 값이 실려 있었으며,
 * `ENERGY_WAVE`(장풍)는 **감지 로직이 없어 발동조차 못 하는데** 사용 가능한 기술로 적혀 있었다.
 *
 * 조작 문서가 틀리면 사용자는 "기능이 고장났다"고 판단한다 — 실제로 그런 보고가 여러 번 있었다.
 * 그래서 수치를 코드에서 뽑아 문서와 대조한다.
 */
const fs = require('fs');
const path = require('path');

const read = (rel) => fs.readFileSync(path.join(__dirname, rel), 'utf8');
const README = read('../README.md');
const CORE = read('../server/static/punch_core.js');
const CLIENT = read('../server/templates/fighter_client.html');
const APP = read('../server/app.py');

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

/** 소스에서 상수 값을 뽑는다 */
function val(src, pattern) {
  const m = src.match(pattern);
  return m ? m[1] : null;
}

/** README 어딘가에 그 숫자가 적혀 있는지 (표기 흔들림 허용: 12.0 == 12, 0.40 == 0.4) */
function docHas(v) {
  if (v === null) return false;
  const forms = new Set([v]);
  const n = Number(v);
  if (!Number.isNaN(n)) {
    forms.add(String(n));                    // 12.0 → 12
    forms.add(n.toFixed(1));                 // 12 → 12.0
    forms.add(n.toFixed(2));
    forms.add(n.toFixed(3));
  }
  for (const f of forms) {
    // 숫자 경계를 지켜 부분 일치(1.6 이 11.6 에 걸리는 것)를 막는다
    const re = new RegExp(`(?<![\\d.])${f.replace('.', '\\.')}(?![\\d])`);
    if (re.test(README)) return true;
  }
  return false;
}

console.log('--- 펀치 판정 임계값 (punch_core.js) ---');
[
  ['PUNCH_ARM',      /PUNCH_ARM:\s*([\d.]+)/],
  ['PUNCH_EXTEND',   /PUNCH_EXTEND:\s*([\d.]+)/],
  ['PUNCH_SPEED',    /PUNCH_SPEED:\s*([\d.]+)/],
  ['PUNCH_REACH_N',  /PUNCH_REACH_N:\s*([\d.]+)/],
  ['PUNCH_GROW_N',   /PUNCH_GROW_N:\s*([\d.]+)/],
  ['PUNCH_WINDOW',   /PUNCH_WINDOW:\s*(\d+)/],
  ['PUNCH_CD',       /PUNCH_CD:\s*(\d+)/],
  ['PUNCH_LOCK',     /PUNCH_LOCK:\s*(\d+)/],
  ['PUNCH_LOCK_MAX', /PUNCH_LOCK_MAX:\s*(\d+)/],
  ['HOOK_VX',        /HOOK_VX:\s*([\d.]+)/],
  ['HOOK_ELBOW',     /HOOK_ELBOW:\s*(\d+)/],
  ['UPPERCUT_VY',    /UPPERCUT_VY:\s*([\d.]+)/],
  ['UPPERCUT_ELBOW', /UPPERCUT_ELBOW:\s*(\d+)/],
].forEach(([name, re]) => {
  const v = val(CORE, re);
  ck(`README 에 ${name} 값이 있다`, docHas(v), `코드 ${v}`);
});

console.log('');
console.log('--- 자세/이동 임계값 (fighter_client.html) ---');
[
  ['PITCH_ON',       /PITCH_ON:\s*([\d.]+)/],
  ['PITCH_OFF',      /PITCH_OFF:\s*([\d.]+)/],
  ['PITCH_BACK_ON',  /PITCH_BACK_ON:\s*([\d.]+)/],
  ['PITCH_BACK_OFF', /PITCH_BACK_OFF:\s*([\d.]+)/],
  ['ROLL_ON',        /ROLL_ON:\s*([\d.]+)/],
  ['ROLL_OFF',       /ROLL_OFF:\s*([\d.]+)/],
  ['ROLL_FLAT',      /ROLL_FLAT:\s*([\d.]+)/],
  ['SHIFT_ON',       /SHIFT_ON:\s*([\d.]+)/],
  ['SHIFT_OFF',      /SHIFT_OFF:\s*([\d.]+)/],
  ['MOVE_SPEED',     /MOVE_SPEED\s*=\s*([\d.]+)/],
  ['AIM_SNAP_RANGE', /AIM_SNAP_RANGE\s*=\s*([\d.]+)/],
  ['ENGAGE_RANGE',   /ENGAGE_RANGE\s*=\s*([\d.]+)/],
  ['ENGAGE_DIST',    /ENGAGE_DIST\s*=\s*([\d.]+)/],
].forEach(([name, re]) => {
  const v = val(CLIENT, re);
  ck(`README 에 ${name} 값이 있다`, docHas(v), `코드 ${v}`);
});

console.log('');
console.log('--- 데미지 표 (app.py) ---');
{
  const specs = {};
  const re = /"(\w+)":\s*\((\d+),\s*([\d.]+),/g;
  let m;
  while ((m = re.exec(APP))) specs[m[1]] = { dmg: Number(m[2]), range: Number(m[3]) };

  for (const k of ['LEFT_JAB', 'RIGHT_CROSS', 'LEFT_HOOK', 'LEFT_UPPERCUT']) {
    ck(`README 에 ${k} 데미지(${specs[k].dmg})가 있다`, docHas(String(specs[k].dmg)));
  }
  // 최강 기술 x 최대 속도 보너스가 10 이하여야 "최소 10대" 주장이 성립한다.
  // **실제로 발동 가능한 기술만** 센다 — ENERGY_WAVE 는 서버에 규격만 있고 감지 로직이 없어
  // 아무도 쓸 수 없으므로, 이걸 최강기로 잡으면 밸런스 주장이 엉뚱하게 실패한다.
  const bonus = Number(val(APP, /min\(velocity,\s*50\.0\)\s*\/\s*(\d+)/));
  // 필살기는 제외한다 — 게이지를 가득 채워야(약 9대 맞아야) 한 번 쓸 수 있는 특수 기술이라
  // "몇 대 버티는가"라는 일반 밸런스와 다른 축에 있다. 넣으면 "2대면 죽는다"가 되어버린다.
  const ULT = 'ENERGY_WAVE';
  const usable = Object.entries(specs)
    .filter(([k]) => k !== ULT && (CORE.includes(`'${k}'`) || CLIENT.includes(`"${k}"`)))
    .map(([, v]) => v.dmg);
  const worst = Math.max(...usable);
  const worstHit = Math.floor(worst * (1 + 50 / bonus));
  ck('"최소 10대는 버틴다" 주장이 수치와 맞는다 (필살기 제외)', Math.floor(100 / worstHit) >= 10,
     `일반 최강 타격 ${worstHit} → ${Math.floor(100 / worstHit)}대`);
  ck('README 가 그 주장을 담고 있다', /최소 10대/.test(README));
  // 필살기가 구현됐다면 문서도 그 사실과 위력을 밝혀야 한다
  if (specs[ULT] && (CORE.includes(`'${ULT}'`) || CLIENT.includes(`"${ULT}"`))) {
    ck('필살기가 구현되면 문서가 "미구현"이라고 하면 안 된다',
       !/ENERGY_WAVE[\s\S]{0,80}(발동할 수 없|미구현)/.test(README));
    ck('README 에 필살기 데미지가 있다', docHas(String(specs[ULT].dmg)),
       `${specs[ULT].dmg}`);
    ck('README 에 분노 게이지 설명이 있다', /분노|RAGE|게이지/.test(README));
  }
}

console.log('');
console.log('--- 구현 상태와 문서가 일치하는가 ---');
{
  // ENERGY_WAVE: 서버에는 있지만 클라이언트에 감지 로직이 없다 → 문서가 그 사실을 밝혀야 한다
  const inServer = /"ENERGY_WAVE"/.test(APP);
  const detectable = /ENERGY_WAVE/.test(CLIENT) || /ENERGY_WAVE/.test(CORE);
  if (inServer && !detectable) {
    ck('ENERGY_WAVE 가 미구현임을 문서가 밝힌다',
       /ENERGY_WAVE[\s\S]{0,80}(발동할 수 없|미구현)/.test(README));
  } else {
    ck('ENERGY_WAVE 감지 로직이 생기면 이 검사를 갱신할 것', detectable === false || inServer);
  }

  // 실제 발동 가능한 액션이 전부 문서에 있는가
  const actions = [];
  const nameRe = /'(LEFT_\w+|RIGHT_\w+)'/g;
  let m;
  while ((m = nameRe.exec(CORE))) if (!actions.includes(m[1])) actions.push(m[1]);
  actions.forEach(a => ck(`README 에 ${a} 가 있다`, README.includes(a)));
  ck('README 에 DUAL_GUARD 가 있다', README.includes('DUAL_GUARD'));

  // 문서에 있는데 코드에 없는 액션은 없어야 한다 (있다면 미구현 표시가 있어야 한다)
  ck('경기 방식이 최후 1인 생존으로 적혀 있다',
     /마지막 1인|최후 1인|LAST MAN STANDING/.test(README));
  ck('제한 시간(60초) 이야기가 남아 있지 않다', !/60초 경기|ROUND 1 :/.test(README));
}

console.log('');
console.log('--- 파일 구조 설명이 실제와 맞는가 ---');
{
  const files = ['punch_core.js', 'face3d.js', 'effects.js', 'humanoid.js'];
  files.forEach(f => {
    const exists = fs.existsSync(path.join(__dirname, '../server/static/', f));
    ck(`${f} 가 존재하고 README 에 언급된다`, exists && README.includes(f),
       exists ? '' : '파일 없음');
  });
}

console.log(fail === 0 ? '\n>>> 전부 통과' : `\n>>> ${fail}개 실패`);
process.exit(fail ? 1 : 0);
