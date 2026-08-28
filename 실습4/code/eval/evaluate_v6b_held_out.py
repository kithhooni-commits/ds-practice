"""evaluate_v6b_held_out.py — v6b(TCN 트리거) 평가. evaluate_video.py --engine tcn_trigger 를
전체 90초에 대해 causal하게 한 번 돌리고(시간 연속성 유지 — 이건 누설이 아니다, 가중치 업데이트가
없는 순전파일 뿐이다), **채점만** train에 쓰인 시간대를 빼고 test-only 구간으로 제한한다.

또한 --engine tcn_trigger 가 정말로 TCN을 썼는지(조용히 실패하지 않았는지) stdout을 직접
확인해 검증한다 — evaluate_video.py 는 이제 로드 실패 시 SystemExit 하므로 비정상 종료
자체가 1차 증거이고, 성공 메시지 문자열 존재를 2차로 재확인한다.
"""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from scoring import load_labels_file, score_predictions

VIDEO_LANDMARKS = SCRIPT_DIR / "video" / "benchmark_landmarks.jsonl"
LABELS_PATH = SCRIPT_DIR / "video" / "benchmark_labels.json"
MODEL_DIR = SCRIPT_DIR.parent / "motion_learning" / "v6b_tcn_trigger"
OUT_DIR = SCRIPT_DIR / "runs" / "v6b_tcn_trigger"
CONFIG_PATH = SCRIPT_DIR / "configs" / "v6b_tcn_trigger.json"

# train_tcn_v6b_trigger.py 와 정확히 같은 시각 — 여기서 어긋나면 "test-only" 주장이 깨진다.
TEST_SINGLETONS_MS = [13133, 18400, 28500, 34000, 45500, 51500]
SINGLETON_RADIUS_MS = 1000
COMBO_BLOCK_MS = (75500, 84000)


def in_test_region(t_ms):
    if COMBO_BLOCK_MS[0] <= t_ms <= COMBO_BLOCK_MS[1]:
        return True
    return any(abs(t_ms - s) <= SINGLETON_RADIUS_MS for s in TEST_SINGLETONS_MS)


def run_engine(engine="tcn_trigger", config_path=CONFIG_PATH, model_dir=MODEL_DIR, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "evaluate_video.py"),
        "--landmarks", str(VIDEO_LANDMARKS),
        "--labels", str(LABELS_PATH),
        "--config", str(config_path),
        "--engine", engine,
        "--tcn-model-dir", str(model_dir),
        "--out-dir", str(out_dir),
    ]
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print("실행:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(f"evaluate_video.py 실패 (code {res.returncode}) — {engine} 엔진이 "
                          f"정상 동작하지 않았다는 뜻. 위 stderr 참고.")

    # --- 검증 1: TCN이 실제로 로드됐는지 stdout에서 확인 ---
    loaded_ok = "🧠 [TCN" in res.stdout and "로드 완료" in res.stdout
    fallback_seen = "로드 실패" in res.stdout
    print("\n" + "=" * 60)
    print(f"[검증] {engine} 엔진이 실제로 TCN을 썼는가?")
    print(f"  - 프로세스 종료 코드: {res.returncode} (0이면 SystemExit 안 걸렸다는 뜻)")
    print(f"  - '🧠 [TCN ... 로드 완료' 로그 존재: {loaded_ok}")
    print(f"  - '로드 실패'(폴백) 로그 존재: {fallback_seen}")
    if not loaded_ok or fallback_seen:
        raise SystemExit("❌ 검증 실패 — TCN이 실제로 로드/사용되지 않았을 가능성이 있다.")
    print("  => ✅ TCN이 정상적으로 로드되어 전체 트리거/분류를 직접 수행했다.")
    print("=" * 60)
    return res.stdout


def load_predictions(csv_path: Path):
    import csv
    punches = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            clean = {k.strip().lstrip("﻿"): v.strip() for k, v in r.items() if k}
            punches.append({
                "t_ms": int(float(clean["t_ms"])),
                "side": clean.get("side", ""),
                "kind": clean.get("kind", ""),
                "action": clean.get("action", ""),
                "conf_margin": float(clean.get("conf_margin", 0) or 0),
            })
    return punches


def main():
    run_engine()

    all_preds = load_predictions(OUT_DIR / "punches.csv")
    all_labels = load_labels_file(LABELS_PATH)

    # --- 전체 90초 기준 (참고용 — leakage-free 아님, train 구간 포함) ---
    full_metrics = score_predictions(all_preds, all_labels)

    # --- test-only (leakage-free) ---
    test_preds = [p for p in all_preds if in_test_region(p["t_ms"])]
    test_truth = [p for p in all_labels["punches"] if in_test_region(p["t_ms"])]
    test_labels = {"tolerance_ms": all_labels["tolerance_ms"], "punches": test_truth}
    test_metrics = score_predictions(test_preds, test_labels)

    train_region_preds = [p for p in all_preds if not in_test_region(p["t_ms"])]

    print("\n" + "=" * 60)
    print("[결과 요약]")
    print(f"전체 90초(참고, 비-leakage-free): 예측 {full_metrics['predicted']}개 / "
          f"정답 {full_metrics['ground_truth']}개 / F1={full_metrics['f1']}")
    print(f"TEST-ONLY(leakage-free, 15개 정답만): 예측 {test_metrics['predicted']}개 / "
          f"정답 {test_metrics['ground_truth']}개 / F1={test_metrics['f1']}")
    print(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']} "
          f"precision={test_metrics['precision']} recall={test_metrics['recall']}")
    print(f"  kind_accuracy={test_metrics['kind_accuracy']} confusion={test_metrics['confusion']}")
    print(f"(train 구간에서 나온 예측 {len(train_region_preds)}개는 test 채점에서 제외됨)")

    report = {
        "version": "v6b_tcn_trigger",
        "description": "TCN이 트리거(언제 나가는지)까지 직접 담당. leakage 방지를 위해 test로 "
                        "지정한 15개 정답(및 ±2500ms)은 학습에서 제외, 채점도 이 test 구간만 사용.",
        "tcn_verification": {
            "returncode_zero": True,
            "load_success_log_found": True,
            "fallback_log_found": False,
            "note": "evaluate_video.py가 --engine tcn_trigger에서 모델 로드 실패 시 SystemExit "
                     "하도록 수정돼 있어, 이 스크립트가 끝까지 돈 것 자체가 1차 증거다.",
        },
        "test_region_definition": {
            "singleton_windows_ms": [[s - SINGLETON_RADIUS_MS, s + SINGLETON_RADIUS_MS] for s in TEST_SINGLETONS_MS],
            "combo_block_ms": list(COMBO_BLOCK_MS),
        },
        "full_90s_reference_NOT_leakage_free": full_metrics,
        "test_only_leakage_free": test_metrics,
        "n_predictions_in_train_region_excluded_from_scoring": len(train_region_preds),
    }
    out_path = OUT_DIR / "metrics_test_only.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] 저장: {out_path}")


if __name__ == "__main__":
    main()
