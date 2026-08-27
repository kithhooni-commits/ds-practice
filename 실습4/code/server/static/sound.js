/**
 * sound.js — 전투 음향 (Web Audio 실시간 합성, 외부 파일 없음)
 *
 *   <script src="/static/sound.js"></script>
 *   const snd = window.createSound();
 *   snd.impact(7, false);   // 타격 — 데미지, 가드 여부
 *   snd.whoosh(28);         // 헛스윙 — 펀치 속도(km/h)
 *   snd.ko();  snd.bell();  snd.step();  snd.hurt();
 *
 * ── 설계 ────────────────────────────────────────────────────────────────────
 * 오디오 파일을 두지 않는다. 저장소가 무거워지고 CDN·경로 문제가 생기며,
 * 무엇보다 **데미지에 따라 소리가 달라져야** 타격감이 산다 — 녹음 파일로는 못 한다.
 *
 * 타격음은 한 겹으로는 "삑" 소리밖에 안 난다. 실제 타격음은 세 성분이 겹친 것이다.
 *   (1) 저역 '퍽'  — 몸통이 울리는 소리. 사인파를 아래로 떨어뜨린다
 *   (2) 중역 클릭  — 뼈·글러브가 부딪히는 단단한 어택
 *   (3) 노이즈 '착' — 가죽이 스치는 질감. 밴드패스로 걸러 짧게
 * 데미지가 클수록 더 낮고 길고 크게 울린다.
 *
 * 브라우저 자동재생 정책상 **사용자 제스처 이후에만** 소리가 난다.
 * 그래서 ensure() 를 클릭/키 입력에 걸어 두고, 그 전에는 조용히 무시한다.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.createSound = api.createSound;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function createSound(opts) {
    const cfg = Object.assign({ master: 0.85 }, opts || {});
    let ctx = null;
    let bus = null;          // 마스터 게인
    let comp = null;         // 컴프레서 — 여러 소리가 겹쳐도 찢어지지 않게
    let noiseBuf = null;
    let muted = false;

    /** AudioContext 는 사용자 제스처 이후에만 살아난다. */
    function ensure() {
      if (muted) return null;
      try {
        if (!ctx) {
          const AC = window.AudioContext || window.webkitAudioContext;
          if (!AC) return null;
          ctx = new AC();
          comp = ctx.createDynamicsCompressor();
          comp.threshold.value = -18;
          comp.ratio.value = 8;
          comp.attack.value = 0.003;
          comp.release.value = 0.18;
          bus = ctx.createGain();
          bus.gain.value = cfg.master;
          bus.connect(comp).connect(ctx.destination);
        }
        if (ctx.state === 'suspended') ctx.resume();
        return ctx.state === 'running' ? ctx : null;
      } catch (e) { return null; }
    }

    /** 화이트 노이즈 버퍼 (한 번만 만들어 재사용) */
    function noise(c) {
      if (noiseBuf) return noiseBuf;
      const n = Math.floor(c.sampleRate * 0.5);
      noiseBuf = c.createBuffer(1, n, c.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      return noiseBuf;
    }

    /** 짧은 게인 엔벨로프 — 지수 감쇠. 0으로 램프하면 예외가 나므로 하한을 둔다. */
    function env(c, node, t, peak, dur) {
      const g = c.createGain();
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(Math.max(0.0002, peak), t + 0.004);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      node.connect(g);
      return g;
    }

    function tone(c, t, type, f0, f1, peak, dur) {
      const o = c.createOscillator();
      o.type = type;
      o.frequency.setValueAtTime(f0, t);
      if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur * 0.9);
      env(c, o, t, peak, dur).connect(bus);
      o.start(t);
      o.stop(t + dur + 0.02);
    }

    function burst(c, t, freq, q, peak, dur, type) {
      const src = c.createBufferSource();
      src.buffer = noise(c);
      src.playbackRate.value = 0.85 + Math.random() * 0.3;
      const f = c.createBiquadFilter();
      f.type = type || 'bandpass';
      f.frequency.value = freq;
      f.Q.value = q;
      src.connect(f);
      env(c, f, t, peak, dur).connect(bus);
      src.start(t);
      src.stop(t + dur + 0.02);
    }

    // ── 타격 ────────────────────────────────────────────────────────────
    /**
     * @param {number} damage  데미지 — 클수록 낮고 길고 크게 울린다
     * @param {boolean} isGuard 가드로 막혔는가 — 둔탁한 저역 대신 단단한 '탁'
     */
    function impact(damage, isGuard) {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      const p = Math.min(1, (damage || 5) / 10);      // 0..1 위력

      if (isGuard) {
        // 가드: 글러브끼리 부딪히는 소리 — 저역이 거의 없고 짧다
        tone(c, t, 'square', 320, 180, 0.16, 0.06);
        burst(c, t, 2400, 1.2, 0.20, 0.07);
        burst(c, t + 0.01, 700, 2.0, 0.10, 0.09);
        return;
      }
      // (1) 저역 '퍽' — 위력이 클수록 낮게 떨어진다
      tone(c, t, 'sine', 165 - p * 55, 42 - p * 12, 0.55 + p * 0.35, 0.16 + p * 0.14);
      // (2) 중역 클릭 — 단단한 어택
      tone(c, t, 'triangle', 520 - p * 120, 190, 0.20 + p * 0.12, 0.055);
      // (3) 노이즈 '착' — 가죽 질감
      burst(c, t, 1500 + p * 900, 0.9, 0.16 + p * 0.14, 0.085);
      // 큰 타격은 꼬리에 울림을 더한다
      if (p > 0.7) burst(c, t + 0.03, 220, 3.5, 0.12, 0.26, 'lowpass');
    }

    /** 헛스윙 — 주먹이 공기를 가르는 소리. 속도(km/h)가 빠를수록 높고 크다. */
    function whoosh(speedKmh) {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      const v = Math.min(1, (speedKmh || 10) / 45);
      const src = c.createBufferSource();
      src.buffer = noise(c);
      const f = c.createBiquadFilter();
      f.type = 'bandpass';
      f.Q.value = 1.1;
      // 필터를 쓸어올렸다 내리면 "슉" 하고 지나가는 느낌이 난다
      f.frequency.setValueAtTime(420 + v * 300, t);
      f.frequency.linearRampToValueAtTime(1500 + v * 1400, t + 0.06);
      f.frequency.linearRampToValueAtTime(500, t + 0.16);
      src.connect(f);
      env(c, f, t, 0.10 + v * 0.16, 0.17).connect(bus);
      src.start(t);
      src.stop(t + 0.2);
    }

    /** 피격 당한 쪽 — 숨이 막히는 저역 + 귀울림 */
    function hurt(damage) {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      const p = Math.min(1, (damage || 5) / 10);
      tone(c, t, 'sine', 120 - p * 30, 55, 0.30 + p * 0.2, 0.22);
      burst(c, t, 400, 1.6, 0.10 + p * 0.1, 0.18, 'lowpass');
      if (p > 0.75) tone(c, t + 0.05, 'sine', 3200, 2600, 0.05, 0.55);   // 이명
    }

    /** K.O. — 낮게 깔리는 임팩트 + 매트에 떨어지는 둔탁한 소리 */
    function ko() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      tone(c, t, 'sine', 140, 34, 0.95, 0.55);
      burst(c, t, 900, 0.8, 0.35, 0.20);
      // 0.35초 뒤 몸이 매트에 닿는 소리
      tone(c, t + 0.35, 'sine', 85, 30, 0.6, 0.45);
      burst(c, t + 0.35, 260, 1.2, 0.28, 0.35, 'lowpass');
    }

    /** 라운드 공 — 3연타 */
    function bell() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      for (let i = 0; i < 3; i++) {
        const tt = t + i * 0.22;
        tone(c, tt, 'square', 880, 860, 0.22, 0.42);
        tone(c, tt, 'sine', 1760, 1740, 0.10, 0.34);
      }
    }

    /** 발소리 — 링 위 스텝 */
    function step() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      burst(c, t, 180 + Math.random() * 90, 1.4, 0.10, 0.07, 'lowpass');
    }

    /** 가드 전개 — 홀로그램 실드가 켜지는 소리 */
    function shield() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      tone(c, t, 'sine', 420, 900, 0.10, 0.16);
      burst(c, t, 3000, 2.5, 0.05, 0.12);
    }

    /** 필살기 — 기를 모았다 터뜨리는 소리. 저역 상승 + 폭발 + 잔향 */
    function ultimate() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      // (1) 차오르는 소리 — 주파수가 올라가며 긴장을 만든다
      const o = c.createOscillator();
      o.type = 'sawtooth';
      o.frequency.setValueAtTime(70, t);
      o.frequency.exponentialRampToValueAtTime(420, t + 0.30);
      const lp = c.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.setValueAtTime(300, t);
      lp.frequency.exponentialRampToValueAtTime(4000, t + 0.30);
      o.connect(lp);
      env(c, lp, t, 0.30, 0.32).connect(bus);
      o.start(t); o.stop(t + 0.36);
      // (2) 방출 — 큰 폭발
      tone(c, t + 0.28, 'sine', 220, 38, 0.95, 0.55);
      burst(c, t + 0.28, 1200, 0.6, 0.45, 0.42);
      // (3) 잔향 — 길게 깔리는 저역
      burst(c, t + 0.34, 300, 1.0, 0.22, 0.70, 'lowpass');
    }

    /** 승리 팡파르 */
    function victory() {
      const c = ensure(); if (!c) return;
      const t = c.currentTime;
      [523, 659, 784, 1047].forEach((f, i) => {
        tone(c, t + i * 0.13, 'triangle', f, f, 0.20, 0.30);
      });
    }

    return {
      ensure, impact, whoosh, hurt, ko, bell, step, shield, victory, ultimate,
      setMuted: (m) => {
        muted = !!m;
        if (bus) bus.gain.value = muted ? 0 : cfg.master;
      },
      isMuted: () => muted,
      setVolume: (v) => { cfg.master = v; if (bus) bus.gain.value = muted ? 0 : v; },
      get context() { return ctx; },
    };
  }

  return { createSound };
});
