#!/usr/bin/env python3
"""E2E Boxing Action Recognition Benchmark Pipeline.

Runs the complete 4-stage pipeline:
  Stage 1: Pose Extraction & Caching (MediaPipe 3D World Landmarks)
  Stage 2: Action Recognition & Kinematics Inference
  Stage 3: Ground Truth Matching & Metrics Scoring (Precision/Recall/F1/Confusion)
  Stage 4: Automated Markdown & HTML Report Generation

Usage:
  python iter3/eval/run_pipeline.py --video iter3/eval/video/benchmark.mp4 --labels iter3/eval/video/benchmark_labels.json
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
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Stage 1 실패:\n{res.stderr}")
        raise RuntimeError("Pose extraction failed")
    print(f"✅ [Stage 1] 랜드마크 추출 완료: {out_jsonl}")


def inference_stage(jsonl_path: Path, out_dir: Path, labels_path: Path = None, config_path: Path = None, annotate_video: Path = None, raw_video: Path = None):
    """Stage 2: Action Detection & Kinematics Engine."""
    print(f"⚙️ [Stage 2] 펀치 키네마틱스 엔진 동작 판정 중...")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_video.py"),
        "--landmarks", str(jsonl_path),
        "--out-dir", str(out_dir)
    ]
    if config_path and config_path.exists():
        cmd.extend(["--config", str(config_path)])
    if labels_path and labels_path.exists():
        cmd.extend(["--labels", str(labels_path)])
    if annotate_video and raw_video:
        cmd.extend(["--annotate", str(annotate_video), str(raw_video)])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Stage 2 실패:\n{res.stderr}")
        raise RuntimeError("Inference failed")
    print(f"✅ [Stage 2] 동작 판정 완료 (out_dir: {out_dir})")


def load_predictions(csv_path: Path):
    """Load punches from output CSV."""
    punches = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean = {k.strip().lstrip("\ufeff"): v.strip() for k, v in r.items() if k}
            punches.append({
                "t_ms": int(float(clean["t_ms"])),
                "frame": int(float(clean.get("frame", 0))),
                "side": clean.get("side", ""),
                "action": clean.get("action", ""),
                "kind": clean.get("action", "").split("_")[-1] if "_" in clean.get("action", "") else clean.get("action", ""),
                "speed_kmh": float(clean.get("speed_kmh", 0)),
                "elbow_deg": float(clean.get("elbow_deg", 0)),
                "conf_margin": float(clean.get("conf_margin", 0)),
            })
    return punches


def calculate_phase_metrics(punches, duration_sec):
    """Calculate False Positives in Rest and Footwork periods."""
    # (t0, t1, phase_name, is_action_phase)
    phases = [
        (0, 6, "1. 준비 (Calibration)", False),
        (6, 18, "2. 직선 펀치 (Straight)", True),
        (18, 23, "⏸ 숨고르기 (Rest 1)", False),
        (23, 35, "3. 훅 펀치 (Hook)", True),
        (35, 40, "⏸ 숨고르기 (Rest 2)", False),
        (40, 52, "4. 어퍼컷 (Uppercut)", True),
        (52, 57, "⏸ 숨고르기 (Rest 3)", False),
        (57, 70, "5. 풋워크 (Footwork)", False),
        (70, 75, "⏸ 숨고르기 (Rest 4)", False),
        (75, 85, "6. 실전 콤보 (Combos)", True),
        (85, 90, "7. 마무리 (Cooldown)", False),
    ]
    rest_fp_count = 0
    footwork_fp_count = 0
    phase_stats = []

    for t0, t1, name, is_action in phases:
        detected = [p for p in punches if t0 * 1000 <= p["t_ms"] < t1 * 1000]
        count = len(detected)
        if not is_action:
            if "풋워크" in name:
                footwork_fp_count += count
            else:
                rest_fp_count += count
        phase_stats.append({
            "phase": name,
            "range": f"{t0:02d}~{t1:02d}s",
            "is_action": is_action,
            "detected": count,
            "punches": [p["action"] for p in detected]
        })

    return {
        "rest_fp": rest_fp_count,
        "footwork_fp": footwork_fp_count,
        "non_action_fp_total": rest_fp_count + footwork_fp_count,
        "phases": phase_stats
    }


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
        status = "🟢 정상" if (p["is_action"] and p["detected"] > 0) or (not p["is_action"] and p["detected"] == 0) else f"⚠️ {p['detected']}회 오검출"
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
    ap.add_argument("--version", default=None, help="Version tag (e.g. v1_baseline, v2_anti_sway)")
    ap.add_argument("--config", default=None, help="Path to version config JSON (e.g. iter3/eval/configs/v2_anti_sway.json)")
    ap.add_argument("--video", default="iter3/eval/video/benchmark.mp4", help="Input video file")
    ap.add_argument("--labels", default="iter3/eval/video/benchmark_labels.json", help="Ground truth labels JSON")
    ap.add_argument("--output-dir", default=None, help="Custom output directory (default: iter3/eval/runs/<version>)")
    ap.add_argument("--force-extract", action="store_true", help="Force re-extract landmarks")
    args = ap.parse_args()

    # 버전 결정
    config_path = Path(args.config).resolve() if args.config else None
    version_tag = args.version
    if not version_tag:
        if config_path:
            version_tag = config_path.stem
        else:
            version_tag = "v1_baseline"

    video_path = Path(args.video).resolve()
    labels_path = Path(args.labels).resolve() if args.labels else None

    # 버전 아카이브 디렉토리
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = SCRIPT_DIR / "runs" / version_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = video_path.with_name(f"{video_path.stem}_landmarks.jsonl")
    csv_path = out_dir / "punches.csv"
    report_md_path = out_dir / "summary_report.md"
    metrics_json_path = out_dir / "metrics.json"

    print("=" * 65)
    print(f"🥊 Boxing Benchmark Pipeline — Version: [{version_tag}]")
    print("=" * 65)
    print(f"• Version Tag : {version_tag}")
    print(f"• Config File : {config_path.name if config_path else '(Default Baseline)'}")
    print(f"• Input Video : {video_path.name}")
    print(f"• Labels File : {labels_path.name if labels_path else 'None'}")
    print(f"• Output Dir  : {out_dir}")
    print("-" * 65)

    start_time = time.time()

    # Stage 1: Pose Extraction
    extract_pose_stage(video_path, jsonl_path, force=args.force_extract)

    # Stage 2: Action Inference
    inference_stage(jsonl_path, out_dir, labels_path=labels_path, config_path=config_path)
    punches = load_predictions(csv_path)

    # Stage 3: Scoring & Metrics
    metrics = {"version": version_tag, "source_video": str(video_path), "predicted_punches": len(punches)}
    if labels_path and labels_path.exists():
        labels = load_labels_file(labels_path)
        metrics = score_predictions(punches, labels)
        metrics["version"] = version_tag
        metrics["source_video"] = str(video_path)

    # Phase FP metrics
    phase_metrics = calculate_phase_metrics(punches, duration_sec=90)
    metrics["phase_analysis"] = phase_metrics

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
