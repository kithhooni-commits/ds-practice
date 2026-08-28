/**
 * tcn_engine.js — motion_learning/train_tcn_real.py 로 학습한 causal TCN을
 * onnxruntime-web으로 브라우저에서 돌리는 실시간 추론 엔진.
 *
 * 중요: 이 엔진은 "펀치가 언제 나가는지(trigger)"를 절대 결정하지 않는다.
 * 그건 항상 punch_core.js(rule-base)의 몫이다 — 속도·뻗음·창(window)·쿨다운 같은
 * 물리 임계값 기반 판정이 노이즈에 훨씬 강하기 때문이다. 이 엔진이 답하는 건 딱 하나,
 * "punch_core.js가 이미 펀치라고 확정한 그 순간, 종류(JAB/HOOK/UPPERCUT 등)가 뭐였는가"뿐이다.
 * (이전 버전은 이 엔진이 자체 디바운스로 발동 자체를 결정했는데, 그 결과 가만히 있어도
 *  모델이 애매하게 흔들리는 것만으로 계속 오검출이 나는 문제가 있었다 — 지금 구조로 바뀌면서
 *  이동/회전/펀치-잠금 관계가 rule-base와 완전히 동일한 코드 경로를 타게 된다.)
 *
 * ENERGY_WAVE는 이 구조에서 트리거할 수 없다 — punch_core.js의 tryPunch는 팔 하나짜리
 * 펀치만 감지하고 양손 동시 동작을 감지하는 트리거가 없기 때문이다. 필요하면 별도의
 * rule 기반 "양손 동시 창" 트리거를 새로 만들고 그 순간의 종류만 이 엔진에 물어보는 식으로
 * 나중에 추가할 수 있다(지금은 범위 밖).
 */
(function (root) {
  'use strict';

  const SEQ_LEN = 60;
  const FEATURE_DIM = 17;
  const CLASSES = [
    'IDLE', 'LEFT_JAB', 'RIGHT_JAB', 'LEFT_HOOK', 'RIGHT_HOOK',
    'LEFT_UPPERCUT', 'RIGHT_UPPERCUT', 'TWO_HAND_GUARD', 'ENERGY_WAVE', 'OTHER',
  ];
  const PUNCH_CLASSES = new Set([
    'LEFT_JAB', 'RIGHT_JAB', 'LEFT_HOOK', 'RIGHT_HOOK', 'LEFT_UPPERCUT', 'RIGHT_UPPERCUT',
  ]);
  // 데이터셋은 오른손 스트레이트를 RIGHT_JAB, 게임(app.py/humanoid.js)은 RIGHT_CROSS라 부른다.
  const ACTION_ALIAS = { RIGHT_JAB: 'RIGHT_CROSS' };

  const KIND_MIN_CONFIDENCE = 0.5; // 이보다 확신 없으면 rule-base의 종류 판정을 그대로 둔다
  const STABLE_HOLD_MS = 200;      // HUD 표시 전용 디바운스 — 발동 판정에는 전혀 관여하지 않는다

  function createTCNEngine() {
    let session = null;
    let scaler = null;
    let ready = false;
    let loadError = null;

    const buffer = []; // Float32Array(17)[] — 오래된 것이 앞, 최신이 뒤 (causal)

    let latest = null;      // 가장 최근 추론 결과 {label, confidence} — 디바운스 없는 raw 값.
                             // 펀치 종류 재판정은 이 값을 그대로 쓴다: punch_core.js가 발동을
                             // 확정한 그 순간에 "지금 모델이 뭐라고 보는지"를 묻는 것이므로,
                             // 여기에 인위적인 지연을 넣으면 정작 그 순간의 판단이 아니게 된다.
    let inferring = false;  // 추론 1건 진행 중이면 다음 push는 건너뛴다 (저사양 기기 보호)

    let candidateLabel = null, candidateSince = 0;
    let stableLabel = null, stableConfidence = 0; // HUD 전용 — 일정 시간 유지돼야 반영

    async function load(modelUrl, scalerUrl) {
      try {
        const [sess, scalerJson] = await Promise.all([
          ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] }),
          fetch(scalerUrl).then((r) => r.json()),
        ]);
        session = sess;
        scaler = scalerJson;
        ready = true;
      } catch (e) {
        loadError = e;
        console.error('[tcn_engine] 모델 로드 실패:', e);
      }
    }

    function buildInputTensor() {
      const n = buffer.length;
      if (n === 0) return null;
      const data = new Float32Array(SEQ_LEN * FEATURE_DIM);
      const { median, scale, clip } = scaler;
      for (let t = 0; t < SEQ_LEN; t++) {
        // 학습 때(real_data._left_pad_causal)와 같은 방식: 짧으면 첫 프레임을 반복해 왼쪽을 채운다.
        const srcIdx = Math.max(0, t - (SEQ_LEN - n));
        const src = buffer[srcIdx];
        for (let f = 0; f < FEATURE_DIM; f++) {
          let v = (src[f] - median[f]) / scale[f];
          if (v > clip) v = clip; else if (v < -clip) v = -clip;
          data[t * FEATURE_DIM + f] = v;
        }
      }
      return data;
    }

    async function runInference(now) {
      inferring = true;
      try {
        const data = buildInputTensor();
        if (!data) return;
        const tensor = new ort.Tensor('float32', data, [1, SEQ_LEN, FEATURE_DIM]);
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];
        const outputs = await session.run({ [inputName]: tensor });
        const logits = outputs[outputName].data;

        let maxLogit = -Infinity;
        for (let i = 0; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
        let sum = 0;
        const exps = new Float64Array(logits.length);
        for (let i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - maxLogit); sum += exps[i]; }
        let bestI = 0, bestP = -1;
        for (let i = 0; i < exps.length; i++) {
          const p = exps[i] / sum;
          if (p > bestP) { bestP = p; bestI = i; }
        }
        latest = { label: CLASSES[bestI], confidence: bestP };

        // HUD 전용 디바운스 갱신. 발동/판정 로직은 절대 이 값을 보지 않는다.
        if (latest.label !== candidateLabel) { candidateLabel = latest.label; candidateSince = now; }
        if (now - candidateSince >= STABLE_HOLD_MS) {
          stableLabel = candidateLabel;
          stableConfidence = latest.confidence;
        }
      } catch (e) {
        console.error('[tcn_engine] 추론 실패:', e);
      } finally {
        inferring = false;
      }
    }

    /** 매 포즈 프레임 호출. 버퍼만 갱신하고, 추론은 백그라운드에서 논블로킹으로 돈다. */
    function push(feat17, now) {
      if (!feat17) return;
      buffer.push(feat17);
      if (buffer.length > SEQ_LEN) buffer.shift();
      if (ready && !inferring) runInference(now); // fire-and-forget — 이전 추론 중이면 이번 프레임은 건너뜀
    }

    /**
     * punch_core.js(rule-base)가 이미 "지금 이 팔이 펀치를 냈다"고 확정한 순간에만 호출한다.
     * 그 팔(side)에 맞는 펀치 라벨을 모델이 충분한 확신으로 내고 있을 때만 종류를 돌려주고,
     * 아니면 null — 호출 쪽은 null이면 rule-base의 classify() 결과를 그대로 쓰면 된다.
     * @param {'L'|'R'} side punch_core.js가 판정한, 실제로 펀치를 낸 팔
     */
    function guessPunchKind(side) {
      if (!latest || latest.confidence < KIND_MIN_CONFIDENCE) return null;
      if (!PUNCH_CLASSES.has(latest.label)) return null;
      const wantSide = side === 'L' ? 'LEFT_' : 'RIGHT_';
      if (!latest.label.startsWith(wantSide)) return null;
      return {
        action: ACTION_ALIAS[latest.label] || latest.label,
        kind: latest.label.split('_').slice(1).join('_'),
        confidence: latest.confidence,
      };
    }

    /** HUD 표시 전용(디바운스됨). 게임 판정에는 절대 쓰지 않는다. */
    function stableState() {
      return stableLabel ? { label: stableLabel, confidence: stableConfidence } : null;
    }

    function reset() {
      buffer.length = 0;
      latest = null;
      candidateLabel = null; candidateSince = 0;
      stableLabel = null; stableConfidence = 0;
    }

    return {
      load, push, guessPunchKind, stableState, reset,
      get ready() { return ready; },
      get loadError() { return loadError; },
    };
  }

  root.createTCNEngine = createTCNEngine;
})(typeof window !== 'undefined' ? window : this);
