/**
 * hybrid_live_check.js — 실제 헤드리스 브라우저로 fighter_client 페이지를 띄워
 *   (1) 엔진 드롭다운이 'tcn'(실측 학습)으로 고정되고 선택 불가능(disabled)한지
 *   (2) TCN 모델(boxing_tcn.onnx)이 실제로 로드되는지(ready===true, loadError===null) —
 *       "TCN 로직에서 에러나서 rule-base로 폴백되는" 이슈가 재발하지 않았는지 확인
 *   (3) localStorage에 다른 모드가 남아 있어도 무시되고 여전히 tcn으로 고정되는지
 * 를 확인한다.
 *
 * 사전조건: python run_arena_server.py --no-ssl --port 8000 (별도 터미널)
 * 실행:     cd iter4/tests && node hybrid_live_check.js http://localhost:8000
 */
const { open, sleep } = require('./_cdp');
const BASE = process.argv[2] || 'http://localhost:8000';

let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

(async () => {
  console.log(`--- 엔진 고정(FIX) 실제 브라우저 점검 (${BASE}) ---`);

  // (0) localStorage에 다른 모드가 남아 있어도 고정값이 이겨야 한다.
  const pre = await open(`${BASE}/client?id=client_1`, { port: 9352, settle: 1500 });
  await pre.evaluate(`localStorage.setItem('motionMode', 'hybrid')`);
  pre.close();

  const f = await open(`${BASE}/client?id=client_1`, { port: 9353, settle: 4000 });
  await sleep(3000); // TCN(onnxruntime-web wasm) 로드 대기

  const st = await f.evaluate(`(() => {
    const sel = document.getElementById('engine-select');
    const options = sel ? Array.from(sel.options).map(o => o.value) : [];
    return {
      hasSelect: !!sel,
      options,
      disabled: sel ? sel.disabled : null,
      selectedValue: sel ? sel.value : null,
      motionMode: motionMode,
      storedBefore: localStorage.getItem('motionMode'), // 위에서 넣은 'hybrid'가 그대로 남아있을 수 있음(무시 확인용)
      tcnReady: tcnEngine.ready,
      tcnLoadError: tcnEngine.loadError ? String(tcnEngine.loadError) : null,
    };
  })()`);

  ck('엔진 드롭다운이 존재한다', st.hasSelect);
  ck("드롭다운 옵션이 'tcn' 하나만 있다", st.options.length === 1 && st.options[0] === 'tcn', JSON.stringify(st.options));
  ck('드롭다운이 비활성화(disabled)되어 있다', st.disabled === true);
  ck("선택값이 'tcn'으로 고정되어 있다", st.selectedValue === 'tcn', st.selectedValue);
  ck("motionMode가 'tcn'으로 고정된다 (localStorage의 'hybrid' 무시)", st.motionMode === 'tcn', st.motionMode);
  ck('TCN 모델이 실제로 로드됐다 (ready===true)', st.tcnReady === true, `loadError=${st.tcnLoadError}`);
  ck('TCN 로드 에러가 없다', st.tcnLoadError === null, st.tcnLoadError);

  // 드롭다운을 강제로 조작해도(개발자도구 등) 실제 판정 로직엔 change 리스너가 안 걸려 있어야 한다.
  const changeAttempt = await f.evaluate(`(() => {
    try {
      document.getElementById('engine-select').disabled = false;
      document.getElementById('engine-select').value = 'rule';
      document.getElementById('engine-select').dispatchEvent(new Event('change'));
      return { threw: false, motionModeAfter: motionMode };
    } catch (e) { return { threw: true, error: String(e) }; }
  })()`);
  ck('change 리스너가 없어 강제 조작해도 motionMode가 안 바뀐다', changeAttempt.motionModeAfter === 'tcn', JSON.stringify(changeAttempt));

  const errs = f.logs.filter(l => l.kind === 'EXCEPTION');
  ck('런타임 예외 0건', errs.length === 0, errs.map(e => e.text).join(' / ') || 'none');

  f.close();
  console.log(fail === 0 ? '\n✅ 전부 통과' : `\n❌ ${fail}건 실패`);
  process.exit(fail === 0 ? 0 : 1);
})().catch((e) => { console.error('하니스 실행 실패:', e); process.exit(1); });
