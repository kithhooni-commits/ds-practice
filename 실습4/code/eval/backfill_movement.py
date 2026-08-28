"""backfill_movement.py — 이미 존재하는 run 디렉터리들에 movement_timeline
과 metrics.json.movement 를 채워 넣는다.

Why:
    run_pipeline.py 로 재실행 가능한 버전(v1~v5)은 이미 movement 를 갖고
    있지만, v5b/v5c/v6/v6b/v6c 는 별도 실험 스크립트로 만들어졌고 각자
    다른 metrics 스키마를 갖는다. 이들을 통째로 재실행하는 건 학습부터
    다시 돌리는 것과 같아서 부적절하다. 반면 movement 통계는 config·엔진
    설정과 무관하게 랜드마크 캐시만 있으면 결정된다 — 그러니 한 번 계산해
    모든 버전에 심어 대시보드에서 무빙 표시가 되게만 하는 게 정공법.

정책 (사용자 결정 2026-08-27, "v0 iter2 만 예외"):
    - v0_iter2_hands: iter2 하드 코드로 뽑힌 결과 (metrics 스키마는 정상
      이지만 소스가 v1~v6 와 실제로는 같다. iter2 시절 코드로 이미
      아카이빙된 결과 자체는 손대지 않는 정책이므로 movement 도 심지
      않는다).
    - 그 외 모든 run 디렉터리: v1_baseline 의 movement_timeline.jsonl 을
      그대로 복사하고, metrics.json 이 존재하면 movement 요약을 in-place
      로 병합한다. metrics.json 이 없는 실험 폴더(v6b/v6c) 는 timeline
      파일만 심는다.

주의:
    - 소스가 benchmark.mp4 랑 다른 실험(예: 별도 held-out 영상)은 이
      스크립트로 처리하면 안 된다. 이번 프로젝트에서는 모든 대상 run 이
      benchmark.mp4 를 소스로 쓴다고 audit 로 확인했다.
    - 실행은 idempotent. 이미 movement_timeline.jsonl 이 있으면 스킵.
      --force 로 강제 덮어쓰기 가능.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from movement_pass import summarize_movement  # noqa: E402
from run_pipeline import DEFAULT_90S_PHASES, load_phase_definitions  # noqa: E402


EXCLUDED = {"v0_iter2_hands"}  # 사용자 정책: iter2 결과는 손대지 않는다


def load_timeline(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_phases(labels_path: Path | None):
    if labels_path is None or not labels_path.exists():
        return DEFAULT_90S_PHASES
    defs = load_phase_definitions(labels_path)
    if defs is None:
        return DEFAULT_90S_PHASES
    if defs == []:
        return None  # unknown case → phase 분석 스킵
    return defs


def merge_movement_into_metrics(metrics_path: Path, movement: dict) -> bool:
    """metrics.json 에 movement 키를 병합. 이미 있으면 덮어쓴다."""
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"    metrics.json 파싱 실패: {exc}")
        return False
    m["movement"] = movement
    metrics_path.write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return True


def update_registry(runs_root: Path, movement: dict, versions: list[str]):
    """registry 의 각 항목에 무빙 스칼라 3개를 채우고, 등록 안 된
    특수 실험은 새 엔트리로 등록한다.

    v1~v5 는 파이프라인 재실행 때 이미 채워졌지만 v5b/v6/v6c 등 특수
    실험은 registry 에 아예 등록되지 않은 경우가 있어 대시보드 셀렉터에
    안 뜬다. 이 함수는 두 가지를 한다:
      1) 이미 등록된 엔트리: 무빙 스칼라 3개를 채운다 (기존 값이 없으면).
      2) 등록 안 된 엔트리: metrics.json (또는 대안으로 metrics_test_only.json)
         에서 f1/tp/fp/fn 을 뽑아 새 엔트리로 추가한다. movement 스칼라도
         함께 채운다.
    """
    registry_path = runs_root / "runs_registry.json"
    if not registry_path.exists():
        return
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    pa = movement.get("phase_analysis") or {}
    fw = pa.get("footwork_recall_proxy")
    sf = pa.get("static_fp_proxy")
    gc = movement.get("guard_coverage_pct")

    registered = {r["version"] for r in reg.get("runs", [])}
    changed = 0
    added = 0

    # (1) 이미 등록된 것: 무빙 스칼라 fill-in
    for r in reg.get("runs", []):
        if r["version"] in versions:
            if r.get("footwork_recall_proxy") is None:
                r["footwork_recall_proxy"] = fw
                changed += 1
            if r.get("static_fp_proxy") is None:
                r["static_fp_proxy"] = sf
                changed += 1
            if r.get("guard_coverage_pct") is None:
                r["guard_coverage_pct"] = gc
                changed += 1

    # (2) 미등록 대상: metrics.json 또는 metrics_test_only.json 로 신규 등록
    for v in versions:
        if v in registered:
            continue
        d = runs_root / v
        # 어느 metrics 파일을 참고할지 결정
        metrics_file = None
        for candidate in ("metrics.json", "metrics_test_only.json"):
            p = d / candidate
            if p.exists():
                metrics_file = p
                break
        if metrics_file is None:
            # metrics 가 아예 없는 폴더는 registry 스칼라를 만들 수 없다.
            # timeline 만 심어두면 대시보드에서 punches.csv 로 최소한 뜨긴 함.
            print(f"  ⚠ {v}: metrics(_test_only).json 없음 → registry 등록 스킵")
            continue
        try:
            m = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ⚠ {v}: {metrics_file.name} 파싱 실패 ({exc})")
            continue

        # metrics_test_only.json 의 경우 정직한 지표는 test_only_leakage_free 안에
        # 있고 최상위 필드는 실험 메타데이터 위주다. 두 스키마를 모두 지원.
        def _pick(m, key, default=None):
            if key in m and m[key] is not None:
                return m[key]
            inner = m.get("test_only_leakage_free") or m.get("full_90s_reference_NOT_leakage_free") or {}
            return inner.get(key, default)

        entry = {
            "version": v,
            "timestamp": "unknown (backfilled)",
            "f1": _pick(m, "f1", 0.0),
            "precision": _pick(m, "precision", 0.0),
            "recall": _pick(m, "recall", 0.0),
            "tp": _pick(m, "tp", 0),
            "fp": _pick(m, "fp", 0),
            "fn": _pick(m, "fn", 0),
            "non_action_fp": (m.get("phase_analysis") or {}).get("non_action_fp_total"),
            "timing_err_ms": _pick(m, "timing_error_ms_mean"),
            "footwork_recall_proxy": fw,
            "static_fp_proxy": sf,
            "guard_coverage_pct": gc,
            "config_path": None,
            "run_dir": str(d),
            "backfilled": True,
        }
        # metrics_test_only 인 경우 표기
        if metrics_file.name == "metrics_test_only.json":
            entry["note"] = "backfilled from metrics_test_only.json (leakage-free test region)"
        reg.setdefault("runs", []).append(entry)
        added += 1

    if changed or added:
        registry_path.write_text(
            json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  📚 registry: {changed} 필드 갱신, {added} 엔트리 추가")


def main():
    ap = argparse.ArgumentParser(description="Backfill movement into existing runs")
    ap.add_argument("--runs-root", default=str(_HERE / "runs"))
    ap.add_argument(
        "--source-timeline",
        default=str(_HERE / "runs" / "v1_baseline" / "movement_timeline.jsonl"),
        help="복사할 원본 movement_timeline.jsonl (기본: v1_baseline)"
    )
    ap.add_argument(
        "--labels",
        default=str(_HERE / "video" / "benchmark_labels.json"),
        help="phase 정의를 읽을 라벨 파일"
    )
    ap.add_argument("--force", action="store_true", help="이미 있어도 덮어쓴다")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    source = Path(args.source_timeline).resolve()
    labels = Path(args.labels).resolve() if args.labels else None

    if not source.exists():
        raise SystemExit(f"source timeline 없음: {source}")

    # movement summary 는 한 번만 계산 (모든 대상이 같은 소스라 동일)
    print(f"[1] source timeline 읽는 중: {source.name}")
    timeline = load_timeline(source)
    print(f"    {len(timeline)} 프레임")

    phases = resolve_phases(labels)
    movement = summarize_movement(timeline, phases=phases)
    pa = movement.get("phase_analysis") or {}
    print(
        f"    footwork_recall_proxy={pa.get('footwork_recall_proxy')}%  "
        f"static_fp_proxy={pa.get('static_fp_proxy')}%  "
        f"guard_coverage={movement.get('guard_coverage_pct')}%"
    )

    # 각 run 처리
    targets = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in EXCLUDED:
            print(f"⏭  {d.name}: 정책상 스킵 (EXCLUDED)")
            continue
        targets.append(d)

    processed = []
    for d in targets:
        print(f"\n▶ {d.name}")
        tl_path = d / "movement_timeline.jsonl"
        if tl_path.exists() and not args.force:
            print(f"    timeline 이미 있음 → 스킵 (--force 로 강제)")
        else:
            shutil.copy2(source, tl_path)
            print(f"    ✅ timeline 복사 ({tl_path.stat().st_size} bytes)")

        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            if merge_movement_into_metrics(metrics_path, movement):
                print(f"    ✅ metrics.json 에 movement 병합")
        else:
            print(f"    ℹ  metrics.json 없음 (특수 실험 폴더) — timeline 만 심음")

        processed.append(d.name)

    # registry 갱신
    print()
    update_registry(runs_root, movement, processed)

    print(f"\n✨ 완료: {len(processed)} 개 run 처리")
    for name in processed:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
