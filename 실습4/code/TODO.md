# CV/AI 기능 개선 TODO

> **목표**: Computer Vision 과제 평가에서 높은 점수를 위해, 기존 게임에 CV/AI 기술을 추가.
> **제약**: 브라우저 실행, FPS 저하 최소화, 실현 가능한 범위.
> **작성일**: 2026-08-24
> **갱신**: 2026-08-25 — 런타임이 MediaPipe **Hands(21×2) → Pose 상체 7노드**로 전환되었습니다.
> 아래 #1·#2는 손 랜드마크를 전제로 작성된 항목이라 입력 규격 재검토가 필요합니다.

---

## 우선순위 요약

| 순위 | 기능 | 예상 시간 | FPS 영향 | CV 어필도 | 상태 |
|---|---|---|---|---|---|
| **1** | LSTM ONNX 브라우저 추론 | 4-6h | 낮음 | ★★★★★ | [ ] |
| **2** | 손 제스처 인식 (특수기) | 2-3h | 없음 | ★★★★ | [ ] |
| **3** | 3D 깊이(Z좌표) 펀치 분류 | 1-2h | 없음 | ★★★ | [x] |
| **4** | 얼굴 사진 → 3D 아바타 머리 | 3-5h | 없음 | ★★★ | [ ] |
| **5** | 웹캠 AR 이펙트 (네온 글로우) | 1-2h | 낮음 | ★★★ | [x] |
| **6** | 바디 세그멘테이션 AR 오버레이 | 4-6h | 중-높음 | ★★★★★ | [ ] |
| **7** | 실제 환경 사진 → 링 배경 | 1-3h | 없음 | ★★ | [x] (곡면 backdrop 방식 · `static/ring_bg.jpg` 배치 필요) |

---

## 1. [최우선] LSTM ONNX 브라우저 실시간 추론

**현재 상태**: BiLSTM 모델 학습 완료(`boxing_lstm.pth`, 합성 데이터 100%), 하지만 **런타임에 미사용**.
클라이언트는 어깨폭 정규화 특징 + 히스테리시스/투표 상태 머신을 사용 중.

> ⚠️ **입력 규격 불일치 (2026-08-25)**: 모델은 `input_dim=63`(손 랜드마크 21개 × 3좌표)로 학습되었으나,
> 런타임은 Pose 상체 7노드로 전환되었습니다. 연동하려면 둘 중 하나가 필요합니다.
> - (권장) `input_dim=21`(7노드 × 3좌표)로 모델·합성 데이터 생성기를 재정의하고 재학습
> - 또는 Pose 노드로부터 63차원 입력을 합성하는 어댑터 작성 (의미상 부자연스러움)
> 재학습 시 world 랜드마크(미터)를 쓰면 좌표 범위 문제(아래 '주의')도 함께 해소됩니다.

**목표**: 학습된 모델을 실제로 브라우저에서 실행하여 6가지 모션을 실시간 분류.

**구현 단계**:
1. PyTorch → ONNX 변환 (Python, 1회성)
   ```python
   dummy = torch.randn(1, 30, 63)  # batch=1, 30프레임, 21랜드마크×3좌표
   torch.onnx.export(model, dummy, 'boxing_lstm.onnx', input_names=['input'], output_names=['output'])
   ```
2. `onnxruntime-web` CDN 추가 (~2MB WASM)
   ```html
   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
   ```
3. 클라이언트에서 30프레임 랜드마크 버퍼 누적 → 2~3Hz로 추론
   ```js
   const session = await ort.InferenceSession.create('/static/boxing_lstm.onnx');
   // 30프레임 모이면 추론 실행 (~2-5ms per call)
   ```
4. 추론 결과로 기존 휴리스틱 대체 (JAB_STRAIGHT, LEFT_HOOK, RIGHT_UPPERCUT 등 6클래스)

**FPS 영향**: 추론 ~2-5ms, 초당 2-3회 실행 → **무시할 수준**

**CV 과제 어필**: ★★★★★
- 데이터 생성 → 학습 → ONNX 변환 → 브라우저 추론의 **End-to-End ML 파이프라인** 완성
- `eval_results.json`의 정확도 수치 (휴리스틱 33.3% → LSTM 100%) 인용 가능
- "Show Numbers" 발표에서 가장 강력한 포인트

**주의**: 합성 데이터로 학습했으므로 실제 MediaPipe 입력과 좌표 범위가 다를 수 있음. 합성 데이터 생성기(`synthetic_boxing_data.py`)를 MediaPipe 좌표 범위([0,1] 정규화)에 맞게 조정하거나, 소량의 실제 랜드마크 데이터를 수집하여 파인튜닝 필요.

---

## 2. 손 제스처 인식 (특수 기술)

**현재 상태**: 런타임이 Pose 상체 7노드로 전환되어 **손가락 랜드마크를 더 이상 받지 않습니다.**

> ⚠️ **선행 결정 필요 (2026-08-25)**: 아래 구현은 Hands의 21개 손가락 랜드마크를 전제로 합니다.
> 도입하려면 Pose와 **Hands를 병용**해야 하고, 과거 두 모델 동시 실행 시 FPS가 30 이하로 떨어진 이력이 있습니다.
> 대안: 특수기 발동을 손가락 모양이 아니라 **상체 포즈**(예: 양팔을 앞으로 모으는 자세)로 재정의하면
> 추가 모델 없이 현재 7노드만으로 구현 가능합니다. ENERGY_WAVE에는 이쪽이 더 잘 맞습니다.

**목표**: 기존 랜드마크 데이터에 기하학적 규칙을 추가하여 특수 제스처 인식.

**구현**: 추가 모델/라이브러리 불필요 — 순수 수학 연산.
```js
function detectGesture(landmarks) {
  const tips = [4, 8, 12, 16, 20];  // 엄지~새끼 끝
  const pips = [3, 6, 10, 14, 18];  // 각 손가락 PIP 관절
  const extended = tips.map((t, i) => {
    if (i === 0) return Math.abs(landmarks[t].x - landmarks[2].x) > 0.05; // 엄지
    return landmarks[t].y < landmarks[pips[i]].y; // 나머지: 끝이 PIP보다 위
  });
  
  if (extended.every(e => e))  return "OPEN_PALM";   // 에너지 실드
  if (extended[0] && !extended[1]) return "THUMBS_UP"; // 도발
  return null;
}
```

**게임 매핑**:
| 제스처 | 인식 조건 | 게임 효과 |
|---|---|---|
| 주먹 쥐기 | 모든 손가락 접힘 | 기본 전투 자세 |
| 손바닥 펴기 | 모든 손가락 펴짐 | 에너지 실드 (방어 강화) |
| 엄지 척 | 엄지만 펴짐 | 도발 (상대에게 이펙트) |
| 검지+중지 | 2개만 펴짐 | 원거리 에너지파 발사 |

**FPS 영향**: **없음** (기존 랜드마크에 산술 연산만 추가, <0.1ms)

**CV 과제 어필**: ★★★★ — 제스처 인식은 CV 고전 주제, 구현이 깔끔하고 데모 효과 좋음

---

## 3. MediaPipe Z좌표 활용 3D 펀치 분류

**현재 상태**: `landmark.x`, `landmark.y`만 사용. **`landmark.z` (깊이)는 무시**.

**목표**: Z좌표의 프레임 간 변화량으로 펀치 유형을 구분.

**구현**: 기존 코드에 z 추적 추가.
```js
// 손목-손끝 z차이 = 팔 뻗은 정도
const depthExtension = lm[0].z - lm[8].z;  // 양수 = 팔이 앞으로 뻗어짐
const depthVelocity = (depthExtension - prevDepthExt) / dt;

// 3D 속도 결합
const vel3D = Math.sqrt(vel2D ** 2 + (depthVelocity * 50) ** 2);
```

**펀치 분류 기준**:
| 모션 | X/Y 움직임 | Z 깊이 변화 |
|---|---|---|
| 잽 (JAB) | 빠른 전방 | 중간 깊이 변화 |
| 크로스 (CROSS) | 크게 전방 | 큰 깊이 변화 |
| 어퍼컷 | 위로 이동 | 중간 깊이 |
| 훅 | 횡으로 이동 | 작은 깊이 |

**FPS 영향**: **없음** (z 데이터는 이미 MediaPipe 결과에 포함, 추가 계산 없음)

**CV 과제 어필**: ★★★ — "2D 입력에서 3D 포즈 추정 활용" 설명 가능

---

## 4. 얼굴 사진 → 3D 아바타 머리 텍스처

**현재 상태**: 아바타 머리는 단색 구(SphereGeometry).

**목표**: 웹캠으로 얼굴 촬영 → 아바타 머리에 텍스처로 매핑.

**구현 접근 (2단계)**:

**Step 1 — 수동 캡처 (기본)**:
```js
// 버튼 클릭 시 웹캠에서 얼굴 영역 캡처
const faceCanvas = document.createElement('canvas');
faceCanvas.width = 256; faceCanvas.height = 256;
faceCtx.drawImage(videoElement, cropX, cropY, cropW, cropH, 0, 0, 256, 256);
const faceTex = new THREE.CanvasTexture(faceCanvas);
humanoid.head.material = new THREE.MeshStandardMaterial({ map: faceTex });
```

**Step 2 — MediaPipe Face Mesh 자동 정렬 (CV 가산점)**:
- `@mediapipe/face_mesh` CDN 추가 (468개 얼굴 랜드마크)
- 코/눈/입 위치 기반으로 얼굴 바운딩 박스 자동 감지 + 정렬
- 기울어진 얼굴도 정규화하여 텍스처 매핑

**FPS 영향**: **없음** (1회성 텍스처 업로드, Face Mesh는 캡처 시에만 실행)

**CV 과제 어필**: ★★★ — Face Mesh 사용 시 얼굴 감지/정렬의 CV 요소 포함

---

## 5. 웹캠 AR 이펙트 (네온 글로우 + 엣지 디텍션)

**현재 상태**: 웹캠 캔버스에 손 관절 오버레이만 표시.

**목표**: CV 기법(엣지 디텍션 등)으로 시각 효과 강화.

**구현 옵션**:

**A) 손 랜드마크 네온 글로우** (1h):
```js
canvasCtx.shadowColor = userColor;
canvasCtx.shadowBlur = 25;
drawLandmarks(canvasCtx, lm, {color:'#fff', fillColor:userColor, radius:5});
```

**B) Sobel 엣지 디텍션 오버레이** (2h):
```js
// 캔버스 픽셀 데이터에 Sobel 커널 적용 → 만화풍 외곽선
const imageData = canvasCtx.getImageData(0, 0, w, h);
// Sobel convolution...
```

**C) 포스터라이즈 (만화 효과)** (0.5h):
```js
canvasCtx.filter = 'contrast(1.5) saturate(2.0)';
```

**FPS 영향**: A/C 없음, B는 낮음

**CV 과제 어필**: ★★★ — Sobel 엣지 디텍션은 CV 기초 기법으로 발표에서 언급하기 좋음

---

## 6. [고급] 바디 세그멘테이션 AR 오버레이

**현재 상태**: 3D 아바타는 프리미티브 메쉬. 실제 플레이어의 모습은 3D 씬에 반영 안 됨.

**목표**: MediaPipe SelfieSegmentation으로 플레이어 실루엣을 추출 → Three.js 빌보드 스프라이트로 렌더.

**구현**:
```html
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js"></script>
```
```js
const segmenter = new SelfieSegmentation({...});
segmenter.onResults((results) => {
  // 배경 제거 → 인물만 추출 → Three.js Sprite로 표시
  offCtx.drawImage(results.segmentationMask, 0, 0);
  offCtx.globalCompositeOperation = 'source-in';
  offCtx.drawImage(results.image, 0, 0);
  spriteTexture.needsUpdate = true;
});
```

**FPS 영향**: **중-높음** (~15-20ms/frame). MediaPipe Hands와 번갈아 실행하여 완화 가능 (세그멘테이션은 ~3fps).

**CV 과제 어필**: ★★★★★ — 실시간 바디 세그멘테이션 + 3D 오버레이 = 진정한 AR. 데모 임팩트 극대.

**주의**: FPS 저하가 클 수 있으므로 마지막에 추가하고, 토글 가능하게 구현 권장.

---

## 7. 실제 환경 사진 → 링 배경

**현재 상태**: 단색 배경 (`scene.background = new THREE.Color(0x07080d)`).

**목표**: 실제 방 사진/파노라마를 3D 링 배경으로 사용.

**구현 (가장 간단)**:
```js
// 휴대폰 파노라마 사진 → equirectangular 매핑
const loader = new THREE.TextureLoader();
loader.load('room_panorama.jpg', (texture) => {
  texture.mapping = THREE.EquirectangularReflectionMapping;
  scene.background = texture;
});
```

**또는 라이브 웹캠 배경 (진짜 AR)**:
```js
const bgTexture = new THREE.CanvasTexture(canvasElement);
// animate 루프에서:
bgTexture.needsUpdate = true;
scene.background = bgTexture;
```

**FPS 영향**: 정적 사진은 없음, 라이브는 낮음

**CV 과제 어필**: ★★ — 기술적으로 간단하지만 시각 효과는 좋음

---

## 권장 구현 순서

**Phase 1 — Quick Wins (3-4h)**: #3 Z좌표 → #2 제스처 → #5 네온 글로우
- FPS 영향 0, 코드 변경 최소, 즉시 데모 가능

**Phase 2 — Core CV Feature (4-6h)**: #1 LSTM ONNX 추론
- 과제 핵심. "학습 → 배포" 파이프라인 완성

**Phase 3 — Visual Polish (3-5h)**: #4 얼굴 텍스처 → #7 환경 배경
- 데모 비주얼 향상

**Phase 4 — Advanced (4-6h, 시간 여유 시)**: #6 바디 세그멘테이션
- FPS 트레이드오프 있으므로 토글 방식으로 구현

---

## 현재 CV/AI 스택 정리

| 기술 | 용도 | 상태 |
|---|---|---|
| MediaPipe Hands | 양손 21개 랜드마크 감지, 펀치 속도 계산, lean 이동 | ✅ 사용 중 |
| MediaPipe Pose | 상체 골격 기울임 추적 | ❌ 제거됨 (FPS 문제) |
| BiLSTM (PyTorch) | 6가지 복싱 모션 분류 | ⚠️ 학습 완료, 런타임 미연동 |
| Three.js | 3D 렌더링 (1인칭 뷰 + arena 뷰) | ✅ 사용 중 |
| WebSocket (FastAPI) | 실시간 4인 동기화 | ✅ 사용 중 |
