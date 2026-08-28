"""evaluate_v6c_held_out.py — v6c: v6b(순수 TCN 트리거, motion_learning/v6b_tcn_trigger)와
**같은 leakage-safe 모델**을 그대로 쓰되, 트리거 조건만 "TCN 확신도 AND 룰베이스 물리조건"으로
바꾼 TCNHybridTriggerEvaluator(--engine tcn_hybrid)로 평가한다.

evaluate_v6b_held_out.py 의 run_engine()/in_test_region() 을 그대로 재사용해, v6b와 **완전히
동일한 leakage 방지·test-only 채점 방식**으로 비교 가능하게 만든다 — 달라지는 건 엔진과 config뿐.
"""
import json
from pathlib import Path

from evaluate_v6b_held_out import (
    run_engine, load_predictions, in_test_region, load_labels_file, score_predictions,
    SCRIPT_DIR, VIDEO_LANDMARKS, LABELS_PATH, MODEL_DIR,
    TEST_SINGLETONS_MS, SINGLETON_RADIUS_MS, COMBO_BLOCK_MS,
)

ENGINE = "tcn_hybrid"
CONFIG_PATH = SCRIPT_DIR / "configs" / "v6c_tcn_hybrid_gate.json"
OUT_DIR = SCRIPT_DIR / "runs" / "v6c_tcn_hybrid_gate"


def main():
    run_engine(engine=ENGINE, config_path=CONFIG_PATH, model_dir=MODEL_DIR, out_dir=OUT_DIR)

    all_preds = load_predictions(OUT_DIR / "punches.csv")
    all_labels = load_labels_file(LABELS_PATH)

    full_metrics = score_predictions(all_preds, all_labels)

    test_preds = [p for p in all_preds if in_test_region(p["t_ms"])]
    test_truth = [p for p in all_labels["punches"] if in_test_region(p["t_ms"])]
    test_labels = {"tolerance_ms": all_labels["tolerance_ms"], "punches": test_truth}
    test_metrics = score_predictions(test_preds, test_labels)

    train_region_preds = [p for p in all_preds if not in_test_region(p["t_ms"])]

    print("\n" + "=" * 60)
    print("[결과 요약] v6c (v6b 모델 + TCN AND rule-physics 게이트)")
    print(f"전체 90초(참고, 비-leakage-free): 예측 {full_metrics['predicted']}개 / "
          f"정답 {full_metrics['ground_truth']}개 / F1={full_metrics['f1']}")
    print(f"TEST-ONLY(leakage-free, {test_metrics['ground_truth']}개 정답만): "
          f"예측 {test_metrics['predicted']}개 / F1={test_metrics['f1']}")
    print(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']} "
          f"precision={test_metrics['precision']} recall={test_metrics['recall']}")
    print(f"  kind_accuracy={test_metrics['kind_accuracy']} confusion={test_metrics['confusion']}")
    print(f"(train 구간에서 나온 예측 {len(train_region_preds)}개는 test 채점에서 제외됨)")

    report = {
        "version": "v6c_tcn_hybrid_gate",
        "description": "v6b(순수 TCN 트리거)와 같은 leakage-safe 모델을 트리거 조건만 "
                        "'TCN 확신도 AND 룰베이스 물리조건'으로 바꿔 재평가.",
        "base_model_dir": str(MODEL_DIR),
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
