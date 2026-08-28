/**
 * heuristic_engine.js — JSON 실측 데이터로 최적화한 순서형 임계값 분류기.
 *
 * punch_core.js가 펀치의 발동 시점과 팔을 정하는 기존 안전 구조는 유지한다.
 * 이 엔진은 TCN과 같은 인터페이스로 최근 heuristic_7j_v1 시퀀스를 보고
 * 이미 발동한 일반 펀치의 종류만 재판정한다.
 */
(function (root) {
  'use strict';

  const FEATURE_DIM = 17;
  const MAX_BUFFER = 60;
  const STABLE_HOLD_MS = 150;
  const KIND_MIN_CONFIDENCE = 0.55;
  const PUNCH_CLASSES = new Set([
    'LEFT_JAB', 'RIGHT_JAB', 'LEFT_HOOK', 'RIGHT_HOOK', 'LEFT_UPPERCUT', 'RIGHT_UPPERCUT',
  ]);
  const ACTION_ALIAS = { RIGHT_JAB: 'RIGHT_CROSS' };
  const I = {
    L_ELBOW: 0, R_ELBOW: 1,
    L_VX: 4, L_VY: 5, L_VZ: 6, R_VX: 7, R_VY: 8, R_VZ: 9,
    L_SPEED: 10, R_SPEED: 11, HANDS_DIST: 12, ELBOW_DIST: 15,
  };

  function clamp01(value) { return Math.max(0, Math.min(1, value)); }

  // 한 프레임/버퍼에서 양손 제스처 조건과 좌우 속도를 함께 계산한다.
  // active side를 고른 뒤 종류 판정에 쓰는 방향·팔꿈치 값은 active side에서만 읽는다.
  function summarize(buffer, lookbackFrames) {
    const n = buffer.length;
    const lookback = Math.max(1, Math.min(Math.round(lookbackFrames || 8), n));
    let maxL = -Infinity, maxR = -Infinity, peakL = 0, peakR = 0;
    let handSum = 0, elbowSum = 0;
    for (let i = 0; i < n; i++) {
      const row = buffer[i];
      if (row[I.L_SPEED] > maxL) { maxL = row[I.L_SPEED]; peakL = i; }
      if (row[I.R_SPEED] > maxR) { maxR = row[I.R_SPEED]; peakR = i; }
      if (i >= n - lookback) { handSum += row[I.HANDS_DIST]; elbowSum += row[I.ELBOW_DIST]; }
    }
    const side = maxL >= maxR ? 'L' : 'R';
    const activeSpeed = side === 'L' ? maxL : maxR;
    const oppositeSpeed = side === 'L' ? maxR : maxL;
    const row = buffer[side === 'L' ? peakL : peakR];
    const offset = side === 'L' ? I.L_VX : I.R_VX;
    const speed = Math.max(row[side === 'L' ? I.L_SPEED : I.R_SPEED], 1e-6);
    const overallMax = Math.max(maxL, maxR), minMax = Math.min(maxL, maxR);
    return {
      handDistLast: handSum / lookback,
      elbowDistLast: elbowSum / lookback,
      overallMax, minMax,
      activeSpeed, oppositeSpeed, activeSide: side,
      balanceRatio: minMax / Math.max(overallMax, 1e-6),
      side,
      lateralRatio: Math.abs(row[offset]) / speed,
      upRatio: -row[offset + 1] / speed,
      forwardRatio: Math.abs(row[offset + 2]) / speed,
      elbowRatio: row[side === 'L' ? I.L_ELBOW : I.R_ELBOW],
    };
  }

  function classify(summary, t) {
    if (summary.minMax > t.wave_min_speed && summary.balanceRatio > t.wave_balance_ratio) {
      return { label: 'ENERGY_WAVE', confidence: clamp01(0.6 + (summary.balanceRatio - t.wave_balance_ratio)) };
    }
    if (summary.handDistLast < t.hand_dist_guard && summary.elbowDistLast < t.elbow_dist_guard) {
      const margin = Math.min(t.hand_dist_guard - summary.handDistLast, t.elbow_dist_guard - summary.elbowDistLast);
      return { label: 'TWO_HAND_GUARD', confidence: clamp01(0.6 + margin) };
    }
    if (summary.overallMax < t.idle_speed_max) {
      return { label: 'IDLE', confidence: clamp01(0.6 + (t.idle_speed_max - summary.overallMax) / Math.max(t.idle_speed_max, 1)) };
    }
    if (summary.balanceRatio > t.other_balance_min) {
      return { label: 'OTHER', confidence: clamp01(0.6 + summary.balanceRatio - t.other_balance_min) };
    }

    const sideName = summary.activeSide === 'L' ? 'LEFT' : 'RIGHT';
    if (summary.upRatio > t.uppercut_vy && summary.elbowRatio < t.uppercut_elbow_ratio) {
      return { label: `${sideName}_UPPERCUT`, confidence: clamp01(0.6 + summary.upRatio - t.uppercut_vy) };
    }
    if (summary.lateralRatio > t.hook_vx && summary.elbowRatio < t.hook_elbow_ratio) {
      return { label: `${sideName}_HOOK`, confidence: clamp01(0.6 + summary.lateralRatio - t.hook_vx) };
    }
    if (summary.forwardRatio >= t.jab_vz_min) {
      return { label: `${sideName}_JAB`, confidence: clamp01(0.6 + summary.forwardRatio - t.jab_vz_min) };
    }
    return { label: 'OTHER', confidence: 0.6 };
  }

  function createOptimizedHeuristicEngine() {
    let thresholds = null, ready = false, loadError = null;
    const buffer = [];
    let latest = null, candidateLabel = null, candidateSince = 0;
    let stableLabel = null, stableConfidence = 0;

    async function load(thresholdUrl) {
      try {
        const response = await fetch(thresholdUrl);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const artifact = await response.json();
        if (artifact.feature_set !== 'heuristic_7j_v1' || !artifact.thresholds) {
          throw new Error('지원하지 않는 휴리스틱 임계값 파일');
        }
        thresholds = artifact.thresholds;
        ready = true;
      } catch (error) {
        loadError = error;
        console.error('[heuristic_engine] 임계값 로드 실패:', error);
      }
    }

    function push(feat17, now) {
      if (!feat17 || feat17.length !== FEATURE_DIM) return;
      buffer.push(feat17);
      if (buffer.length > MAX_BUFFER) buffer.shift();
      if (!ready || buffer.length < 3) return;
      latest = classify(summarize(buffer, thresholds.lookback_frames), thresholds);
      if (latest.label !== candidateLabel) { candidateLabel = latest.label; candidateSince = now; }
      if (now - candidateSince >= STABLE_HOLD_MS) {
        stableLabel = candidateLabel;
        stableConfidence = latest.confidence;
      }
    }

    function guessPunchKind(side) {
      if (!latest || latest.confidence < KIND_MIN_CONFIDENCE || !PUNCH_CLASSES.has(latest.label)) return null;
      const sidePrefix = side === 'L' ? 'LEFT_' : 'RIGHT_';
      if (!latest.label.startsWith(sidePrefix)) return null;
      return {
        action: ACTION_ALIAS[latest.label] || latest.label,
        kind: latest.label.split('_').slice(1).join('_'),
        confidence: latest.confidence,
      };
    }

    function stableState() {
      return stableLabel ? { label: stableLabel, confidence: stableConfidence } : null;
    }

    function reset() {
      buffer.length = 0; latest = null;
      candidateLabel = null; candidateSince = 0;
      stableLabel = null; stableConfidence = 0;
    }

    return {
      load, push, guessPunchKind, stableState, reset,
      get ready() { return ready; },
      get loadError() { return loadError; },
    };
  }

  root.createOptimizedHeuristicEngine = createOptimizedHeuristicEngine;
})(typeof window !== 'undefined' ? window : this);
