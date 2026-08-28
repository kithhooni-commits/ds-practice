/**
 * paint_harness.js — 머리 바깥 덮기가 **얼굴을 잡아먹지 않는지** 텍스처 픽셀로 잰다.
 *
 * 서버가 떠 있어야 한다:  python run_arena_server.py
 *   cd tests && node paint_harness.js [베이스URL]
 *
 * 실제로 겪은 버그: 랜드마크는 카메라 **전체 프레임** 기준 0~1 인데 텍스처는 얼굴만
 * 잘라낸 사진이다. 좌표계를 맞추지 않으면 머리 타원이 실제의 1/4 크기로 잡혀
 * 얼굴 대부분이 머리카락색에 덮인다 — 눈·코만 남고 나머지가 까맣게 나왔다.
 * UV 좌표를 세는 검사로는 절대 안 잡힌다. 텍스처 픽셀을 직접 봐야 한다.
 */
const { open } = require('./_cdp');
const BASE = process.argv[2] || 'https://localhost:8000';
const LAB = `(() => {
  const N = 468, IW = 640, IH = 480;
  // 카메라 전체 프레임 기준 랜드마크 (얼굴은 프레임의 일부만 차지)
  const lm = [];
  for (let i = 0; i < N; i++) {
    const a = i / N * Math.PI * 2, r = 0.10 + 0.03 * Math.sin(i * 2.7);
    lm.push({ x: 0.52 + Math.cos(a) * r * 0.78, y: 0.46 + Math.sin(a) * r, z: -0.05 * Math.cos(a) });
  }
  const OV = (typeof FACEMESH_FACE_OVAL !== 'undefined') ? FACEMESH_FACE_OVAL : null;
  if (OV) {
    const seen = new Set(); OV.forEach(e => { seen.add(e[0]); seen.add(e[1]); });
    const ids = [...seen];
    ids.forEach((id, k) => {
      const a = k / ids.length * Math.PI * 2;
      lm[id] = { x: 0.52 + Math.cos(a) * 0.115, y: 0.46 + Math.sin(a) * 0.15, z: 0.02 };
    });
  }
  // 얼굴 주변만 잘라낸 512x512 텍스처 (fighter_client 가 하는 것과 동일)
  let x0=1,y0=1,x1=0,y1=0;
  lm.forEach(q => { x0=Math.min(x0,q.x); x1=Math.max(x1,q.x); y0=Math.min(y0,q.y); y1=Math.max(y1,q.y); });
  const mx=(x1-x0)*0.22, my=(y1-y0)*0.18;
  const crop = { x0:(x0-mx)*IW, y0:(y0-my)*IH, w:(x1-x0+mx*2)*IW, h:(y1-y0+my*2)*IH };

  // 사진: 잘라낸 영역 전체를 밝은 살구색으로 (얼굴이 살아 있으면 밝게 남아야 한다)
  const c = document.createElement('canvas'); c.width=512; c.height=512;
  const g = c.getContext('2d');
  g.fillStyle = 'rgb(235,190,160)'; g.fillRect(0,0,512,512);
  g.fillStyle = 'rgb(40,26,18)'; g.fillRect(0,0,512,70);   // 위쪽은 머리카락

  const img = new Image();
  return new Promise(res => {
    img.onload = () => {
      const f = window.createFace3D({ landmarks: lm, uvLandmarks: lm, image: img,
        width: 2.6, aspect: IW/IH, crop, imageW: IW, imageH: IH });
      if (!f) return res(JSON.stringify({ err:'null' }));
      // 실제로 쓰이는 텍스처를 읽어 밝은 픽셀 비율을 센다
      const t = f.mesh.material.map.image;
      const cc = document.createElement('canvas'); cc.width=128; cc.height=128;
      const gg = cc.getContext('2d'); gg.drawImage(t, 0, 0, 128, 128);
      const d = gg.getImageData(0,0,128,128).data;
      let bright=0, dark=0;
      for (let i=0;i<d.length;i+=4) {
        const lum = d[i]*0.299 + d[i+1]*0.587 + d[i+2]*0.114;
        if (lum > 120) bright++; else dark++;
      }
      res(JSON.stringify({ 밝은픽셀: bright, 어두운픽셀: dark,
                           얼굴비율: +(bright/(bright+dark)*100).toFixed(1) }));
    };
    img.onerror = () => res(JSON.stringify({ err:'img' }));
    img.src = c.toDataURL('image/png');
  });
})()`;
let fail = 0;
const ck = (n, c, x) => {
  console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${x !== undefined ? `  (${x})` : ''}`);
  if (!c) fail++;
};

(async () => {
  console.log('--- 머리 바깥 덮기가 얼굴을 남기는가 ---');
  const a = await open(`${BASE}/arena`, { port: 9393, settle: 7000, fakeMedia: false });
  const r = JSON.parse(await a.evaluate(LAB));
  await a.close();

  ck('얼굴이 만들어진다', !r.err, r.err || 'ok');
  // 잘라낸 사진은 대부분이 얼굴이다. 절반 이상 남아야 정상 —
  // 좌표계가 어긋나면 15% 아래로 떨어진다.
  ck('덮기 후에도 얼굴이 남는다', !r.err && r.얼굴비율 >= 45,
     r.err ? '-' : `밝은 영역 ${r.얼굴비율}%`);
  ck('전부 덮이지는 않는다', !r.err && r.밝은픽셀 > 3000,
     r.err ? '-' : `${r.밝은픽셀}px`);

  console.log(fail === 0 ? String.fromCharCode(10) + '>>> 전부 통과'
                         : String.fromCharCode(10) + '>>> ' + fail + '개 실패');
  process.exit(fail ? 1 : 0);
})();
