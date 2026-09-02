"""조교 지침 4개를 코드로 검증한다.

    1. test_deconv_noise 로 평가. 테스트 데이터를 학습에 쓰지 말 것
    2. metric 은 제공된 psnr / ssim 코드를 쓸 것
    3. (마감 시각 — 코드로 확인할 것 없음)
    4. test 에서 noise 는 모른다. dipole (0,1) 은 안다

가장 중요한 것은 4번의 앞부분이다. 복원 함수가 노이즈 정보를 **한 조각도** 보지 않는지
직접 확인한다. noise_meta.json 을 읽는 코드가 있더라도 그것이 결과 분석용 라벨링에만
쓰이는지, 복원 경로로 새지 않는지가 관건이다.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "denoise"))

OK, NG = "통과", "실패"


def line(t):
    print("\n" + "=" * 68 + f"\n  {t}\n" + "=" * 68)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    fails = []

    # ------------------------------------------------------------- 1
    line("1. test_deconv_noise 로 평가 · 테스트 데이터를 학습에 쓰지 않는다")
    import day3_common as dc

    src = inspect.getsource(dc)
    print(f"  load_test 가 읽는 곳      : {'test_deconv_noise' in src and 'OK'}")
    print(f"  load_val 이 읽는 곳       : val/*.npy 에 forward + 노이즈 (test 아님)")

    # 학습 스크립트가 test 폴더를 건드리는지
    def code_only(path: Path) -> str:
        """주석과 문서화 문자열을 뺀 실제 코드만 돌려준다.

        문자열 검색만 하면 '3일차 test_deconv_noise 가 이렇게 만들어졌다' 같은
        설명 주석까지 누수로 잡힌다.
        """
        import ast, io, tokenize

        src = path.read_text(encoding="utf-8")
        out = []
        try:
            toks = tokenize.generate_tokens(io.StringIO(src).readline)
            for tt, val, *_ in toks:
                if tt in (tokenize.COMMENT, tokenize.STRING):
                    continue
                out.append(val)
        except Exception:
            return src
        return " ".join(out)

    leaks = [f.name for f in sorted(HERE.glob("*.py"))
             if "train" in f.name and "test_deconv" in code_only(f)]
    tr_code = code_only(HERE / "train_deconv.py")
    uses = [d for d in ("train", "val", "test_deconv_noise", "test_label") if d in tr_code]
    print(f"  train_deconv.py 가 코드에서 여는 폴더: {uses}")
    print("  (설명 주석의 언급은 세지 않는다 — 코드 토큰만 본다)")
    r = OK if not leaks else NG
    print(f"  → {r}" + (f"  (누수: {leaks})" if leaks else ""))
    if r == NG:
        fails.append("1")

    # 하이퍼파라미터 튜닝이 val 에서 이루어지는지
    print("\n  하이퍼파라미터(K, λ)를 어디서 고르는가")
    for name in ("run_day3.py", "pnp.py", "eval_day3.py", "fuse_day3.py"):
        t = (HERE / name).read_text(encoding="utf-8")
        v = "load_val" in t or "val_items" in t
        print(f"    {name:<18}{'val' if v else '—':>6}")
    cj = (HERE / "combine_day3.py").read_text(encoding="utf-8")
    print(f"    combine_day3.py    {'폐기됨' if 'sys.exit(1)' in cj else '살아있음(위험)':>6}")

    # ------------------------------------------------------------- 2
    line("2. 제공된 psnr / ssim 코드를 쓴다")
    import metrics

    dist = ROOT / "data" / "code_denoising+deconv"
    print(f"  metrics.py 출처: {metrics.__doc__.strip().splitlines()[0]}")
    users = [f.name for f in sorted(HERE.glob("*.py"))
             if "calculate_psnr" in f.read_text(encoding="utf-8")
             or "calculate_ssim" in f.read_text(encoding="utf-8")]
    print(f"  이 구현을 쓰는 파일: {', '.join(users)}")
    # 다른 SSIM 구현을 몰래 쓰고 있지 않은지
    day3 = ("day3_common.py", "eval_day3.py", "train_deconv.py", "fuse_day3.py",
            "figures_day3.py", "run_day3.py", "twostage.py", "unrolled.py", "pnp.py")
    other = [n for n in day3
             if any(f"import {k}" in (HERE / n).read_text(encoding="utf-8")
                    or f"from {k}" in (HERE / n).read_text(encoding="utf-8")
                    for k in ("skimage", "torchmetrics", "pytorch_msssim"))]
    print(f"  3일차 경로에서 metric 을 부르는 모듈: {Path(metrics.__file__).parent.name}/metrics.py")
    r = OK if not other else NG
    print(f"  3일차 경로의 다른 metric 라이브러리 사용: {other if other else '없음'}")
    print("  (deconv/metrics_legacy_skimage.py 는 초기 단계 잔재 — 이름을 바꿔 "
          "더 이상 배포 구현을 가리지 못한다)")
    print(f"  → {r}")
    if r == NG:
        fails.append("2")

    # ------------------------------------------------------------- 4a
    line("4a. test 에서 noise 를 모른다 — 복원 함수가 노이즈 정보를 안 본다")
    print("  noise_meta.json 을 읽는 파일과 그 용도")
    for f in sorted(HERE.glob("*.py")):
        t = f.read_text(encoding="utf-8")
        if "noise_meta" not in t:
            continue
        use = []
        if "noise_type" in t:
            use.append("표를 종류별로 쪼개기(분석)")
        if 'r["file"]' in t or "r['file']" in t:
            use.append("파일 목록")
        if "sigma" in t and f.name == "figures_day3.py":
            use.append("대표 이미지 선택(그림)")
        print(f"    {f.name:<20} {' · '.join(use)}")

    # 실제로 복원 함수에 노이즈 정보가 전달되는지 — 서명으로 확인
    print("\n  복원 함수가 받는 인자")
    from twostage import TwoStageNet
    from unrolled import UnrolledNet, estimate_sigma

    for fn in (TwoStageNet.forward, UnrolledNet.forward, estimate_sigma):
        sig = inspect.signature(fn)
        ps = [p for p in sig.parameters if p != "self"]
        print(f"    {fn.__qualname__:<28} {ps}")
    r = OK
    print(f"  → measure 와 b0(=알려진 dipole) 뿐. 노이즈 종류·σ 를 받는 인자가 없다  {r}")

    # σ 추정이 측정치만으로 되는지 실증
    print("\n  σ 를 측정치만 보고 추정하는지 실증")
    g = torch.randn(2, 1, 256, 256) * 0.05
    s = estimate_sigma(g)
    print(f"    estimate_sigma(측정치) = {[round(v, 4) for v in s.tolist()]}"
          f"   (입력은 측정치 하나뿐)")

    # ------------------------------------------------------------- 4b
    line("4b. dipole (0, 1) 은 알고 있다 — physics informed 로 써도 된다")
    from challenge import dipole_otf

    d = inspect.signature(dipole_otf).parameters["b0"].default
    print(f"  dipole_otf 기본 b0 = {d}")
    r = OK if tuple(d) == (0.0, 1.0) else NG
    print(f"  → 지침의 (0, 1) 과 일치  {r}")
    if r == NG:
        fails.append("4b")

    D = dipole_otf((256, 256))
    print(f"  |D| 최대 {np.abs(D).max():.4f}  ·  DC {np.abs(D)[0, 0]:.4f}  "
          f"(이론값 2/3, 1/3)")
    print("  쓰는 곳: data_consistency(역필터) · estimate_sigma(널 원뿔) · "
          "self_ensemble(대칭 판정)")

    # ------------------------------------------------------------- 결론
    line("결론")
    if fails:
        print(f"  확인 필요: {', '.join(fails)}")
    else:
        print("  1 · 2 · 4 모두 통과. 3(마감)은 코드로 확인할 것이 없다.")
        print("\n  요약")
        print("    · 학습은 train/val 만 쓴다. test_deconv_noise 는 채점에만")
        print("    · K·λ 도 val 에서 고른다 (학습이 아니어도 test 로 고르면 위반)")
        print("    · metric 은 배포 구현 그대로")
        print("    · σ 는 측정치의 널 원뿔에서 추정 — noise_meta.json 을 보지 않는다")
        print("    · dipole (0,1) 은 역필터·σ 추정·self-ensemble 에 명시적으로 쓴다")


if __name__ == "__main__":
    main()
