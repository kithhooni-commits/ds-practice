#!/usr/bin/env python3
"""E2E Boxing Action Recognition Benchmark Pipeline.

Runs the complete 4-stage pipeline:
  Stage 1: Pose Extraction & Caching (MediaPipe 3D World Landmarks)
  Stage 2: Action Recognition & Kinematics Inference
  Stage 3: Ground Truth Matching & Metrics Scoring (Precision/Recall/F1/Confusion)
  Stage 4: Automated Markdown & HTML Report Generation

Usage:
  python iter4/eval/run_pipeline.py --video iter4/eval/video/benchmark.mp4 --labels iter4/eval/video/benchmark_labels.json
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# extract_landmarks 는 cv2/mediapipe 를 끌고 온다. Stage 1 은 랜드마크 캐시가 있으면
# 통째로 건너뛰는데, 여기서 import 하면 캐시가 있어도 CV 스택 없이는 파이프라인이 시작조차 못 한다.
try:
    from extract_landmarks import DEFAULT_MODEL as POSE_MODEL
except ImportError:
    POSE_MODEL = SCRIPT_DIR / "models" / "pose_landmarker_full.task"
from scoring import load_labels_file, score_predictions


def extract_pose_stage(video_path: Path, out_jsonl: Path, force: bool = False):
    """Stage 1: MediaPipe 3D Pose Extraction with caching."""
    if out_jsonl.exists() and not force:
        print(f"⏩ [Stage 1] 랜드마크 캐시 발견: {out_jsonl.name} (추출 생략)")
        return

    print(f"⚙️ [Stage 1] MediaPipe 3D 랜드마크 추출 중... ({video_path.name})")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "extract_landmarks.py"),
        str(video_path),
        "-o", str(out_jsonl),
        "--model", str(POSE_MODEL)
    ]
    # capture_output=True 로 stdout 을 삼키면 TCN 로드 실패·랜드마크 진행률 같은
    # 진단 메시지가 사용자에게 도달하지 않는다 (v4/v5 가 사실은 rule 결과였는데
    # "완료"로 표기됐던 은폐 사고의 원인). stdout 은 실시간으로 그대로 흘려보낸다.
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Pose extraction failed (return code {res.returncode})")
    print(f"✅ [Stage 1] 랜드마크 추출 완료: {out_jsonl}")


def inference_stage(jsonl_path: Path, out_dir: Path, labels_path: Path = None, config_path: Path = None, engine: str = "rule", annotate_video: Path = None, raw_video: Path = None, tcn_model_dir: Path = None):
    """Stage 2: Action Detection & Kinematics Engine."""
    print(f"⚙️ [Stage 2] 펀치 엔진 동작 판정 중... (Engine: {engine.upper()})")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_video.py"),
        "--landmarks", str(jsonl_path),
        "--out-dir", str(out_dir),
        "--engine", engine
    ]
    if config_path and config_path.exists():
        cmd.extend(["--config", str(config_path)])
    if labels_path and labels_path.exists():
        cmd.extend(["--labels", str(labels_path)])
    if annotate_video and raw_video:
        cmd.extend(["--annotate", str(annotate_video), str(raw_video)])
    if tcn_model_dir:
        cmd.extend(["--tcn-model-dir", str(tcn_model_dir)])

    # stdout 을 실시간으로 흘려야 TCN 로드/폴백 로그가 보인다.
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Inference failed (return code {res.returncode})")
    print(f"✅ [Stage 2] 동작 판정 완료 (Engine: {engine.upper()}, out_dir: {out_dir})")


def load_predictions(csv_path: Path):
    """Load punches from output CSV."""
    punches = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean = {k.strip().lstrip("\ufeff"): v.strip() for k, v in r.items() if k}
            action = clean.get("action", "")
            raw_kind = clean.get("kind", "")
            if not raw_kind:
                suffix = action.split("_")[-1] if "_" in action else action
                raw_kind = suffix
            
            # Label dictionary normalization: JAB/CROSS/STRAIGHT -> STRAIGHT
            if raw_kind.upper() in ("JAB", "CROSS", "STRAIGHT"):
                kind = "STRAIGHT"
            elif "HOOK" in raw_kind.upper():
                kind = "HOOK"
            elif "UPPERCUT" in raw_kind.upper() or "UPPER" in raw_kind.upper():
                kind = "UPPERCUT"
            else:
                kind = raw_kind.upper()

            punches.append({
                "t_ms": int(float(clean["t_ms"])),
                "frame": int(float(clean.get("frame", 0))),
                "side": clean.get("side", ""),
                "action": action,
                "kind": kind,
                "speed_kmh": float(clean.get("speed_kmh", 0)),
                "elbow_deg": float(clean.get("elbow_deg", 0)),
                "conf_margin": float(clean.get("conf_margin", 0)),
            })
    return punches


# 90초 프로토콜 기본 phase 정의. calculate_phase_metrics 와 movement 요약이
# 공유한다. 예전에는 calculate_phase_metrics 안에만 있어서 movement 쪽이
# phase_defs=None 을 받으면 phase_analysis 를 스킵해버리는 버그가 있었다.
DEFAULT_90S_PHASES = [
    (0, 6, "1. 준비 (Calibration)", False),
    (6, 19, "2. 직선 펀치 (Straight)", True),
    (19, 23, "⏸ 숨고르기 (Rest 1)", False),
    (23, 35, "3. 훅 펀치 (Hook)", True),
    (35, 40, "⏸ 숨고르기 (Rest 2)", False),
    (40, 52, "4. 어퍼컷 (Uppercut)", True),
    (52, 57, "⏸ 숨고르기 (Rest 3)", False),
    (57, 70, "5. 풋워크 (Footwork)", False),
    (70, 75, "⏸ 숨고르기 (Rest 4)", False),
    (75, 85, "6. 실전 콤보 (Combos)", True),
    (85, 90, "7. 마무리 (Cooldown)", False),
]


def calculate_phase_metrics(punches, duration_sec, matched_pred_indices=None, phases=None):
    """구간별 검출 통계와 비동작 구간의 순수 오검출(FP) 수를 계산한다.

    이전 버전 문제:
      * phase 2 경계가 6~18s 였는데 라벨 마지막 크로스는 18400ms 라서
        정상 검출이 phase 3("Rest 1", 18~23s) 로 흘러 non_action_fp 로 잡혔다.
      * TP 매칭 여부와 무관하게 "비동작 구간에 잡힌 모든 예측"을 오검출로
        세서, tolerance 로 이미 정답과 매칭된 예측이 이중으로 페널티를 받았다.

    개선:
      * phase 정의를 인자로 받아 라벨 파일에서 주입할 수 있게 한다.
        (예전엔 90초 프로토콜이 함수 본문에 하드코딩되어 다른 벤치마크를
         평가하면 phase_analysis 가 조용히 오염됐다.)
      * matched_pred_indices 가 주어지면 그 인덱스의 예측은 TP 이므로
        non_action_fp 에서 제외한다. t_ms 대신 인덱스로 판정해 좌·우 팔이
        같은 프레임에서 동시 발화 등으로 중복 t_ms 가 생겨도 오염되지 않는다.
    """
    matched_set = set(matched_pred_indices or [])

    if phases is None:
        # 기본값: iter4/eval/video/benchmark_90s_protocol.
        # 라벨 실제 t_ms 를 감싸도록 조정: straight 마지막 = 18400ms → phase 2
        # 종료를 19s 로, 첫 훅 = 24500ms → phase 3 시작은 23s 유지.
        phases = DEFAULT_90S_PHASES

    rest_fp_count = 0
    footwork_fp_count = 0
    phase_stats = []

    # 인덱스로 매칭 여부를 판단하려면 원래 순서를 유지한 채로 순회해야 한다.
    # punches 는 CSV 로드 시점 순서(=시간순) 로 이미 정렬돼 있다.
    for t0, t1, name, is_action in phases:
        detected_pairs = [
            (i, p) for i, p in enumerate(punches)
            if t0 * 1000 <= p["t_ms"] < t1 * 1000
        ]
        detected = [p for _, p in detected_pairs]
        count = len(detected)
        # 비동작 구간이라도 TP 매칭된 예측은 오검출이 아니다.
        # 인덱스 기반이라 같은 t_ms 예측이 여럿 있어도 정확히 그 중 매칭된
        # 것만 TP 로 잡힌다.
        fp_here = [p for i, p in detected_pairs if i not in matched_set]
        fp_count = len(fp_here)
        if not is_action:
            if "풋워크" in name:
                footwork_fp_count += fp_count
            else:
                rest_fp_count += fp_count
        phase_stats.append({
            "phase": name,
            "range": f"{t0:02d}~{t1:02d}s",
            "is_action": is_action,
            "detected": count,
            "fp": fp_count if not is_action else 0,
            "punches": [p["action"] for p in detected]
        })

    return {
        "rest_fp": rest_fp_count,
        "footwork_fp": footwork_fp_count,
        "non_action_fp_total": rest_fp_count + footwork_fp_count,
        "phases": phase_stats
    }


def load_phase_definitions(labels_path: Path):
    """labels.json 에서 phases 정의를 읽는다. 없으면 None.

    스키마:
      {"phases": [{"t0": 0, "t1": 6, "name": "Calibration", "is_action": false}, ...]}
    또는 case 이름이 "benchmark_90s_protocol" 이면 기본 90초 프로토콜을 사용한다.
    """
    if not labels_path or not labels_path.exists():
        return None
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw_phases = data.get("phases")
    if raw_phases:
        return [
            (int(p["t0"]), int(p["t1"]), str(p["name"]), bool(p["is_action"]))
            for p in raw_phases
        ]

    # 명시적 phase 정의가 없고 알려지지 않은 case 면 phase 분석을 스킵할 수
    # 있도록 sentinel 리턴. 90초 프로토콜만 하드코딩 기본값을 쓴다.
    case = data.get("case", "")
    if case == "benchmark_90s_protocol":
        return None  # 기본값 사용
    # 다른 case 는 안전을 위해 phase 정보를 명시하도록 유도.
    print(f"⚠️ [Phase] labels 파일에 phases 정의가 없고 case='{case}' 는 알려진 프로토콜이 아닙니다. phase_analysis 를 스킵합니다.")
    return []  # 빈 리스트 = phase 분석 스킵


def generate_markdown_report(metrics: dict, phase_metrics: dict, punches: list, output_path: Path):
    """Generate professional executive summary report in Markdown."""
    tp, fp, fn = metrics.get("tp", 0), metrics.get("fp", 0), metrics.get("fn", 0)
    precision = metrics.get("precision", 0) or 0.0
    recall = metrics.get("recall", 0) or 0.0
    f1 = metrics.get("f1", 0) or 0.0
    kind_acc = metrics.get("kind_accuracy", 0) or 0.0
    timing_err = metrics.get("timing_error_ms_mean", 0) or 0.0

    md = f"""# 🥊 Boxing Action Recognition Benchmark Report

**생성 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**평가 대상**: `{metrics.get('source_video', 'Benchmark Video')}`  
**허용 오차 윈도우**: `±{metrics.get('tolerance_ms', 400)}ms`

---

## 1. 📊 종합 정량 성적표 (Overall Benchmark Score)

| 핵심 지표 (Metric) | 측정 수치 | 판정 기준 (Target) | 달성 여부 |
| :--- | :---: | :---: | :---: |
| **F1-Score (종합)** | **{f1:.4f}** | $\\ge 0.850$ | {'🟢 PASS' if f1 >= 0.85 else '🟡 REVIEW'} |
| **Precision (정밀도)** | **{precision:.4f}** ({tp}/{tp+fp}) | $\\ge 0.800$ | {'🟢 PASS' if precision >= 0.80 else '🟡 REVIEW'} |
| **Recall (재현율)** | **{recall:.4f}** ({tp}/{tp+fn}) | $\\ge 0.850$ | {'🟢 PASS' if recall >= 0.85 else '🟡 REVIEW'} |
| **종류 분류 정확도 (Kind Acc)** | **{kind_acc*100:.1f}%** | $\\ge 90.0\\%$ | {'🟢 PASS' if kind_acc >= 0.90 else '🟡 REVIEW'} |
| **비동작/풋워크 오검출 (Non-Action FP)** | **{phase_metrics['non_action_fp_total']}회** | $\\le 3회$ | {'🟢 PASS' if phase_metrics['non_action_fp_total'] <= 3 else '🔴 과검출 발생'} |
| **평균 타격 지연 시간 (Timing MAE)** | **{timing_err:.1f} ms** | $\\le 50.0 ms$ | {'🟢 PASS' if timing_err <= 50 else '🟡 REVIEW'} |

---

## 2. 📋 90초 프로토콜 구간별 검출 상세 (Phase Breakdown)

| 프로토콜 구간 | 시간대 | 성격 | 검출 횟수 | 상태 |
| :--- | :---: | :---: | :---: | :---: |
"""
    for p in phase_metrics["phases"]:
        # 상태 판정 로직: 동작 구간은 검출이 있으면 정상, 비동작 구간은 fp(TP 제외)
        # 가 0 이면 정상. 예전에는 detected==0 만 봤기 때문에 라벨이 걸쳐 있는
        # 구간의 정상 검출까지 "오검출"로 표시되던 문제가 있었다.
        if p["is_action"]:
            status = "🟢 정상" if p["detected"] > 0 else "⚠️ 미검출"
        else:
            fp_here = p.get("fp", p["detected"])
            status = "🟢 정상" if fp_here == 0 else f"⚠️ {fp_here}회 오검출"
        md += f"| **{p['phase']}** | `{p['range']}` | {'동작' if p['is_action'] else '휴식/준비'} | **{p['detected']}회** | {status} |\n"

    md += """
---

## 3. 🎯 펀치 종류별 혼동 행렬 (Confusion Matrix)

```text
"""
    confusion = metrics.get("confusion", {})
    if confusion:
        for k, v in sorted(confusion.items()):
            md += f"  • {k:24s} : {v}회\n"
    else:
        md += "  (매칭된 정답 데이터 없음)\n"
    md += f"""```

---

## 4. 💡 주요 분석 및 개선 제안 (Actionable Insights)

1. **비동작 구간(풋워크/휴식) 과검출 방지**:
   * 현재 풋워크/휴식 구간에서 총 **{phase_metrics['non_action_fp_total']}회**의 오검출이 발생했습니다.
   * `punch_core.js`의 `PUNCH_SPEED` 임계값을 `1.6 m/s -> 1.8 m/s`로 상향하거나, 상체 롤링 중 펀치 감도를 억제하는 `TILT_SUPPRESSION`을 강화하면 해결됩니다.
2. **연타 콤보 및 훅 회수 시 중복 검출 방지**:
   * 펀치 회수(Retract) 시 팔꿈치가 빠르게 당겨지는 과정이 잽으로 오인식되는 문제를 방지하기 위해 `PUNCH_EXTEND (0.40)` 가드를 높이는 것을 권장합니다.

---
*Report generated automatically by Antigravity Benchmark Pipeline.*
"""
    output_path.write_text(md, encoding="utf-8")
    print(f"📄 [Stage 4] 마크다운 보고서 작성 완료: {output_path}")


def update_runs_registry(runs_dir: Path, version_tag: str, metrics: dict, phase_metrics: dict, config_path: Path):
    """Update global runs registry index."""
    registry_path = runs_dir.parent / "runs_registry.json"
    registry = {"runs": []}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {"runs": []}

    tp, fp, fn = metrics.get("tp", 0), metrics.get("fp", 0), metrics.get("fn", 0)
    # movement 지표는 있으면 실린다. 없으면 (초기 버전 재실행 전) None.
    movement = metrics.get("movement") or {}
    mv_pa = movement.get("phase_analysis") or {}
    entry = {
        "version": version_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "f1": metrics.get("f1", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "non_action_fp": phase_metrics.get("non_action_fp_total", 0),
        "timing_err_ms": metrics.get("timing_error_ms_mean", 0.0),
        "footwork_recall_proxy": mv_pa.get("footwork_recall_proxy"),
        "static_fp_proxy": mv_pa.get("static_fp_proxy"),
        "guard_coverage_pct": movement.get("guard_coverage_pct"),
        "config_path": str(config_path) if config_path else None,
        "run_dir": str(runs_dir)
    }

    # 기존 버전 엔트리 업데이트 또는 신규 추가
    existing_idx = next((i for i, r in enumerate(registry["runs"]) if r["version"] == version_tag), None)
    if existing_idx is not None:
        registry["runs"][existing_idx] = entry
    else:
        registry["runs"].append(entry)

    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📚 [Registry] 버전 아카이브 인덱스 갱신: {registry_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Version-Controlled End-to-End Boxing Benchmark Pipeline")
    ap.add_argument("--version", default=None, help="Version tag (e.g. v1_baseline, v4_tcn_hybrid)")
    ap.add_argument("--engine", default="rule", choices=["rule", "tcn", "tcn_trigger", "tcn_hybrid"], help="Punch classification engine (rule, tcn, tcn_trigger, or tcn_hybrid)")
    ap.add_argument("--tcn-model-dir", default=None, help="Override dir containing boxing_tcn.pth + boxing_tcn_scaler.json")
    ap.add_argument("--config", default=None, help="Path to version config JSON (e.g. iter4/eval/configs/v4_tcn_hybrid.json)")
    ap.add_argument("--video", default=str(SCRIPT_DIR / "video" / "benchmark.mp4"), help="Input video file")
    ap.add_argument("--labels", default=str(SCRIPT_DIR / "video" / "benchmark_labels.json"), help="Ground truth labels JSON")
    ap.add_argument("--output-dir", default=None, help="Custom output directory (default: iter4/eval/runs/<version>)")
    ap.add_argument("--force-extract", action="store_true", help="Force re-extract landmarks")
    ap.add_argument("--overwrite", action="store_true", help="Explicitly allow overwriting an existing version run")
    args = ap.parse_args()

    # 버전 결정
    config_path = Path(args.config).resolve() if args.config else None
    version_tag = args.version
    if not version_tag:
        if config_path:
            version_tag = config_path.stem
        else:
            version_tag = "v1_baseline" if args.engine == "rule" else "v4_tcn_hybrid"

    video_path = Path(args.video).resolve()
    labels_path = Path(args.labels).resolve() if args.labels else None

    # 버전 아카이브 디렉토리
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = SCRIPT_DIR / "runs" / version_tag

    # 🔒 불변성 보호 가드 (기존 평가 버전 임의 덮어쓰기 방지)
    if (out_dir / "metrics.json").exists() and not args.overwrite:
        print(f"🛑 [보호 가드] 버전 '{version_tag}'의 평가 결과가 이미 존재합니다 ({out_dir}).")
        print("   과거 이터레이션 결과의 불변성을 유지하기 위해 덮어쓰기를 중단합니다.")
        print(f"   새로운 버전을 평가하려면 `--version v4_...`와 같이 새로운 버전명을 지정하세요.")
        print("   (불가피하게 재평가하려면 `--overwrite` 옵션을 추가하세요.)")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = video_path.with_name(f"{video_path.stem}_landmarks.jsonl")
    csv_path = out_dir / "punches.csv"
    report_md_path = out_dir / "summary_report.md"
    metrics_json_path = out_dir / "metrics.json"

    print("=" * 65)
    print(f"🥊 Boxing Benchmark Pipeline — Version: [{version_tag}]")
    print("=" * 65)
    print(f"• Version Tag : {version_tag}")
    print(f"• Engine Type : {args.engine.upper()} ({'Causal TCN Deep Learning' if args.engine == 'tcn' else 'Rule-based Kinematics'})")
    print(f"• Config File : {config_path.name if config_path else '(Default)'}")
    print(f"• Input Video : {video_path.name}")
    print(f"• Labels File : {labels_path.name if labels_path else 'None'}")
    print(f"• Output Dir  : {out_dir}")
    print("-" * 65)

    start_time = time.time()

    # Stage 1: Pose Extraction
    extract_pose_stage(video_path, jsonl_path, force=args.force_extract)

    # Stage 2: Action Inference
    tcn_model_dir = Path(args.tcn_model_dir).resolve() if args.tcn_model_dir else None
    inference_stage(jsonl_path, out_dir, labels_path=labels_path, config_path=config_path, engine=args.engine, tcn_model_dir=tcn_model_dir)
    punches = load_predictions(csv_path)

    # Stage 2.5: Movement / Rotation / Guard 상태 판정
    # 펀치는 이벤트, 무빙은 프레임 상태다. 채점 파이프라인이 원래 벤치마크에는
    # 무빙을 표시하지 못했는데 (benchmark_labels.json 은 punches 만 갖는다),
    # 대시보드에 "펀치 + 무빙 + 가드" 를 함께 뿌리려면 랜드마크 캐시에서 무빙
    # timeline 을 한 번 더 뽑아야 한다. FullActionEvaluator 는 이미 punch_core.js
    # 의 TUNE 과 동기화된 상태로 존재하므로 그대로 재사용.
    try:
        from movement_pass import run_movement_pass, summarize_movement
        movement_timeline_path = out_dir / "movement_timeline.jsonl"
        print("⚙️ [Stage 2.5] 무빙/회전/가드 상태 판정 중...")
        movement_timeline = run_movement_pass(jsonl_path, movement_timeline_path)
        print(f"✅ [Stage 2.5] 무빙 timeline {len(movement_timeline)} 프레임 → {movement_timeline_path.name}")
    except Exception as exc:
        # 무빙 pass 는 부가 산출물이므로 실패해도 파이프라인은 계속 간다.
        # (라벨/펀치 채점은 이 실패와 무관하다.)
        print(f"⚠️ [Stage 2.5] 무빙 pass 실패 (무시하고 진행): {exc}")
        movement_timeline = []

    # Stage 3: Scoring & Metrics
    metrics = {"version": version_tag, "source_video": str(video_path), "predicted_punches": len(punches)}
    if labels_path and labels_path.exists():
        labels = load_labels_file(labels_path)
        metrics = score_predictions(punches, labels)
        metrics["version"] = version_tag
        metrics["source_video"] = str(video_path)

    # Phase FP metrics — TP 매칭된 예측은 non_action_fp 에서 제외한다.
    # pred_index 로 판정해 같은 t_ms 예측이 여러 개여도 정확히 분리된다.
    matched_indices = [m["pred_index"] for m in metrics.get("matches", [])]
    # phase 정의는 labels 파일에서 우선적으로 읽되, 없으면 90초 프로토콜 기본값.
    # 알려지지 않은 case 면 phase 분석을 스킵해 registry 오염을 막는다.
    phase_defs = load_phase_definitions(labels_path) if labels_path else None
    if phase_defs == []:
        # 명시적으로 스킵
        phase_metrics = {
            "rest_fp": 0,
            "footwork_fp": 0,
            "non_action_fp_total": 0,
            "phases": [],
            "skipped": True,
            "reason": "no phase definition for this labels case",
        }
    else:
        phase_metrics = calculate_phase_metrics(
            punches, duration_sec=90,
            matched_pred_indices=matched_indices,
            phases=phase_defs,
        )
    metrics["phase_analysis"] = phase_metrics

    # movement 요약 — phase_defs 가 None(=labels 파일에 phases 필드 없음, case
    # 는 알려진 90초 프로토콜) 이면 DEFAULT_90S_PHASES 를 그대로 쓴다. 빈
    # 리스트(=알려지지 않은 case) 면 phase_analysis 없이 전체 통계만 저장.
    if movement_timeline:
        try:
            if phase_defs is None:
                mv_phases = DEFAULT_90S_PHASES
            elif phase_defs == []:
                mv_phases = None  # 명시적 스킵
            else:
                mv_phases = phase_defs
            movement_summary = summarize_movement(movement_timeline, phases=mv_phases)
        except Exception as exc:
            print(f"⚠️ [Stage 2.5] movement 요약 실패: {exc}")
            movement_summary = {"error": str(exc)}
        metrics["movement"] = movement_summary
        # 콘솔에 눈에 띄는 두 지표만 출력
        pa = movement_summary.get("phase_analysis") or {}
        fw = pa.get("footwork_recall_proxy")
        sf = pa.get("static_fp_proxy")
        gc = movement_summary.get("guard_coverage_pct")
        if fw is not None or sf is not None:
            print(f"🚶 [Stage 2.5] footwork_recall_proxy={fw}%  static_fp_proxy={sf}%  guard_coverage={gc}%")

    # Save metrics JSON
    metrics_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 [Stage 3] 메트릭 JSON 저장 완료: {metrics_json_path}")

    # Stage 4: Generate Markdown Report
    generate_markdown_report(metrics, phase_metrics, punches, report_md_path)

    # Update Registry
    update_runs_registry(out_dir, version_tag, metrics, phase_metrics, config_path)

    # Copy latest run to legacy output/benchmark for instant /eval dashboard sync
    legacy_out = SCRIPT_DIR / "output" / "benchmark"
    legacy_out.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(csv_path, legacy_out / "punches.csv")
        shutil.copy2(metrics_json_path, legacy_out / "metrics.json")
    except Exception:
        pass

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"✨ [{version_tag}] 파이프라인 실행 완료! (총 소요시간: {elapsed:.2f}초)")
    print(f"• 요약 보고서: {report_md_path}")
    print(f"• 수치 메트릭: {metrics_json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
