/**
 * motion_features.js — 실시간 MediaPipe 랜드마크로 heuristic_7j_v1 근사 피처(17차원)를 뽑는다.
 *
 * motion_learning/collected_pose 의 학습 데이터에 저장된 heuristic_7j_v1 값을 역산해 맞춘 것이다.
 * 각도·거리·비율(9개 채널)은 실측 데이터와 소수점까지 정확히 일치하는 공식을 확인했다:
 *   - 정규화 기준: 어깨 2D(가로세로만, z 제외) 폭
 *   - elbow_angle_ratio = angleDeg(shoulder, elbow, wrist) / 180
 *   - reach / hands_distance / wrist_to_nose / elbow_distance = 3D 유클리드 거리 / 어깨 2D 폭
 *
 * 속도·평균 z(8개 채널)는 원본 파이프라인의 정확한 평활 방식을 알아낼 소스가 없어(수집 도구가
 * 이 저장소에 없음) "직전 프레임과의 차분 / dt / 어깨폭"으로 근사했다 — 스케일과 부호는 맞지만
 * 학습 데이터와 완전히 같은 수치는 아니다. boxing_tcn_scaler.json 의 robust scaling + clip이
 * 이 근사 오차를 어느 정도 흡수하지만, 실제 웹캠으로 확인 후 필요하면 캘리브레이션할 것.
 *
 * 입력: MediaPipe Pose 의 poseLandmarks (정규화 이미지 좌표, x/y/z/visibility)
 *       — fighter_client.html 의 P.{NOSE,L_SH,R_SH,L_EL,R_EL,L_WR,R_WR} 인덱스와 동일 랜드마크 셋.
 */
(function (root) {
  'use strict';

  const FEATURE_NAMES = [
    'left_elbow_angle_ratio', 'right_elbow_angle_ratio', 'left_reach', 'right_reach',
    'left_wrist_vx', 'left_wrist_vy', 'left_wrist_vz',
    'right_wrist_vx', 'right_wrist_vy', 'right_wrist_vz',
    'left_wrist_speed', 'right_wrist_speed', 'hands_distance',
    'left_wrist_to_nose', 'right_wrist_to_nose', 'elbow_distance', 'average_wrist_z',
  ];

  function dist3(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  function angleDeg(a, b, c) {
    // punch_core.js 와 동일 공식(내각, deg) — 있으면 그걸 재사용해 한 곳에서만 정의되게 한다.
    if (root.PunchCore && root.PunchCore.angleDeg) return root.PunchCore.angleDeg(a, b, c);
    const abx = a.x - b.x, aby = a.y - b.y, abz = a.z - b.z;
    const cbx = c.x - b.x, cby = c.y - b.y, cbz = c.z - b.z;
    const d = Math.hypot(abx, aby, abz) * Math.hypot(cbx, cby, cbz);
    if (d < 1e-6) return 180;
    const cos = (abx * cbx + aby * cby + abz * cbz) / d;
    return Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
  }

  function createMotionFeatures() {
    let prevL = null, prevR = null, prevT = 0;

    /**
     * @param {object} P 랜드마크 인덱스 맵 {NOSE,L_SH,R_SH,L_EL,R_EL,L_WR,R_WR}
     * @param {Array} lm MediaPipe poseLandmarks
     * @param {number} now performance.now() (ms)
     * @returns {Float32Array(17)|null} wristOK 가 아니면 null
     */
    function step(P, lm, now) {
      const nose = lm[P.NOSE], lsh = lm[P.L_SH], rsh = lm[P.R_SH];
      const lel = lm[P.L_EL], rel = lm[P.R_EL], lwr = lm[P.L_WR], rwr = lm[P.R_WR];
      if (!nose || !lsh || !rsh || !lel || !rel || !lwr || !rwr) return null;

      const sh2d = Math.max(Math.hypot(lsh.x - rsh.x, lsh.y - rsh.y), 1e-3);

      const lElbowRatio = angleDeg(lsh, lel, lwr) / 180;
      const rElbowRatio = angleDeg(rsh, rel, rwr) / 180;
      const lReach = dist3(lwr, lsh) / sh2d;
      const rReach = dist3(rwr, rsh) / sh2d;
      const handsDist = dist3(lwr, rwr) / sh2d;
      const lWristToNose = dist3(lwr, nose) / sh2d;
      const rWristToNose = dist3(rwr, nose) / sh2d;
      const elbowDist = dist3(lel, rel) / sh2d;
      const avgWristZ = (lwr.z + rwr.z) / 2 / sh2d;

      let lvx = 0, lvy = 0, lvz = 0, rvx = 0, rvy = 0, rvz = 0;
      const dt = prevT ? (now - prevT) / 1000 : 0;
      if (prevL && dt > 0.008 && dt < 0.4) {
        lvx = (lwr.x - prevL.x) / dt / sh2d;
        lvy = (lwr.y - prevL.y) / dt / sh2d;
        lvz = (lwr.z - prevL.z) / dt / sh2d;
        rvx = (rwr.x - prevR.x) / dt / sh2d;
        rvy = (rwr.y - prevR.y) / dt / sh2d;
        rvz = (rwr.z - prevR.z) / dt / sh2d;
      }
      prevL = { x: lwr.x, y: lwr.y, z: lwr.z };
      prevR = { x: rwr.x, y: rwr.y, z: rwr.z };
      prevT = now;

      const lSpeed = Math.hypot(lvx, lvy, lvz);
      const rSpeed = Math.hypot(rvx, rvy, rvz);

      return new Float32Array([
        lElbowRatio, rElbowRatio, lReach, rReach,
        lvx, lvy, lvz, rvx, rvy, rvz,
        lSpeed, rSpeed, handsDist, lWristToNose, rWristToNose, elbowDist, avgWristZ,
      ]);
    }

    function reset() { prevL = null; prevR = null; prevT = 0; }

    return { step, reset, FEATURE_NAMES };
  }

  root.createMotionFeatures = createMotionFeatures;
})(typeof window !== 'undefined' ? window : this);
