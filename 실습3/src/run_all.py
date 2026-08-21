"""Drive TI + DreamBooth-LoRA(/DoRA/prior-preservation) training/generation/evaluation
across concepts, one at a time (a single GPU must run these sequentially). Safe to
re-run: skips any step whose output file already exists, so a crash partway through
just resumes.

    python run_all.py                                   # all 10 concepts, plain LoRA, 4-bit local GPU
    python run_all.py --no_4bit --skip pet_cat5          # Colab/H100: skip what's already done locally
    python run_all.py --only pet_cat5 --variants lora,dora,lora_prior
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CLASS_PROMPT = {  # kept in sync with evaluation.py's CLASS_PROMPT
    "pet_cat5": "cat",
    "actionfigure_2": "action figure",
    "decoritems_woodenpot": "wooden pot",
    "furniture_sofa2": "sofa",
    "instrument_music2": "guitar",
    "luggage_backpack1": "backpack",
    "person_3": "person",
    "scene_waterfall": "waterfall",
    "transport_tank": "tank",
    "wearable_jacket1": "jacket",
}

# textual-inversion initializer_token must resolve to a single vocab token;
# multi-word class nouns need a single-word stand-in for initialization only.
TI_INIT_TOKEN = {
    "actionfigure_2": "figure",
    "decoritems_woodenpot": "pot",
}

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
RESULTS_LOG = os.path.join("results", "scores.log")
failures = []


def variant_args(variant, concept, class_noun):
    if variant == "lora":
        return []
    if variant == "dora":
        return ["--use_dora"]
    if variant == "lora_prior":
        return ["--with_prior_preservation",
                "--class_data_dir", f"class_images/{concept}",
                "--class_prompt", f"a photo of a {class_noun}",
                "--num_class_images", "50"]
    if variant == "dora_prior":
        return ["--use_dora", "--with_prior_preservation",
                "--class_data_dir", f"class_images/{concept}",
                "--class_prompt", f"a photo of a {class_noun}",
                "--num_class_images", "50"]
    raise ValueError(f"unknown variant {variant!r}")


def run(cmd, log_path):
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"exit {result.returncode}: {' '.join(cmd)} (see {log_path})")


def safe_run(cmd, log_path, step_name):
    t0 = time.time()
    try:
        run(cmd, log_path)
        print(f"  [ok] {step_name} ({time.time() - t0:.0f}s)", flush=True)
        return True
    except Exception as e:
        print(f"  [FAILED] {step_name}: {e}", flush=True)
        failures.append(step_name)
        return False


def wait_for(path, timeout_s=3 * 3600, poll=30):
    waited = 0
    while not os.path.exists(path):
        time.sleep(poll)
        waited += poll
        if waited >= timeout_s:
            raise RuntimeError(f"timed out waiting for {path}")


def eval_and_log(concept, method, images_dir):
    if not os.path.exists(os.path.join(images_dir, "9.png")):
        print(f"  [skip eval] {concept}/{method}: no generated images", flush=True)
        return
    r = subprocess.run(
        [PY, "evaluation.py", "--dataset", "dataset", "--prompts", "prompts",
         "--concept", concept, "--images", images_dir],
        capture_output=True, text=True)
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== {concept} / {method} ===\n{r.stdout}{r.stderr}\n")
    print(r.stdout.strip(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated concept subset (default: all)")
    ap.add_argument("--skip", help="comma-separated concepts to skip")
    ap.add_argument("--variants", default="lora",
                     help="comma-separated: lora,dora,lora_prior,dora_prior")
    ap.add_argument("--max_train_steps", default="300")
    ap.add_argument("--no_4bit", action="store_true", help="passthrough to the training scripts (Colab/H100)")
    ap.add_argument("--best_of_n", default="4",
                     help="candidates per prompt for dreambooth (lora/dora/*_prior) generation, CLIP-best kept; TI unaffected")
    ap.add_argument("--gen_suffix", default="",
                     help="suffix for dreambooth output dirs, e.g. --gen_suffix _bo1 writes "
                          "generated/<concept>_<variant>_bo1. Lets an inference-only ablation "
                          "(different --best_of_n) reuse trained weights without overwriting "
                          "the existing images")
    ap.add_argument("--skip_ti", action="store_true",
                     help="skip TI entirely (it ignores --best_of_n, so ablations on it are no-ops)")
    ap.add_argument("--extra_train_args", default="",
                     help="extra flags forwarded verbatim to train_dreambooth_lora.py, e.g. "
                          "--extra_train_args '--center_crop --no_flip'. Pair with --train_tag "
                          "so the run gets its own weight/image dirs")
    ap.add_argument("--trigger_token", default="zwx",
                     help="rare token the subject is bound to. DreamBooth popularised "
                          "'sks', but SD3.5's text encoders read that as the SKS rifle: "
                          "it puts a rifle in the frame for every concept (a rifle "
                          "floats in front of scene_waterfall, and actionfigure_2's "
                          "robe-and-wand figure comes out in combat fatigues). Any "
                          "genuinely meaningless string works better")
    ap.add_argument("--train_tag", default="",
                     help="suffix for BOTH the weight dir and the generated dir, e.g. _ccnoflip "
                          "-> db_lora_<concept>_ccnoflip + generated/<concept>_lora_ccnoflip. "
                          "Keeps a training ablation from overwriting the main run")
    args = ap.parse_args()

    variants = args.variants.split(",")
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    ordered = ["pet_cat5"] + [c for c in CLASS_PROMPT if c != "pet_cat5"]
    concepts = [c for c in ordered if c not in skip and (only is None or c in only)]
    common_flags = ["--max_train_steps", args.max_train_steps] + (["--no_4bit"] if args.no_4bit else [])

    for concept in concepts:
        class_noun = CLASS_PROMPT[concept]
        init_token = TI_INIT_TOKEN.get(concept, class_noun)
        placeholder = f"<{concept}>"
        trigger = f"{args.trigger_token} {class_noun}"

        log_path = os.path.join("logs", f"{concept}.log")
        ti_dir = f"ti_{concept}"
        ti_embeds = os.path.join(ti_dir, "learned_embeds.bin")
        gen_ti_dir = os.path.join("generated", f"{concept}_ti")

        print(f"\n===== {concept} ({class_noun}) =====", flush=True)

        if args.skip_ti:
            print("  [skip] TI (--skip_ti)", flush=True)
        elif os.path.exists(ti_embeds):
            print(f"  [skip] TI already trained -> {ti_embeds}", flush=True)
        elif concept == "pet_cat5" and not args.no_4bit:
            print("  waiting for the in-progress pet_cat5 TI training to finish...", flush=True)
            wait_for(ti_embeds)
            print("  [ok] pet_cat5 TI training finished", flush=True)
        else:
            safe_run(
                [PY, "train_textual_inversion.py",
                 "--data_dir", f"dataset/{concept}", "--output_dir", ti_dir,
                 "--placeholder_token", placeholder, "--initializer_token", init_token] + common_flags,
                log_path, f"{concept}: TI train")

        if not args.skip_ti:
            if os.path.exists(ti_embeds) and not os.path.exists(os.path.join(gen_ti_dir, "9.png")):
                safe_run(
                    [PY, "generate.py", "--method", "ti", "--embeds_path", ti_embeds,
                     "--concept", concept, "--output_dir", gen_ti_dir],
                    log_path, f"{concept}: TI generate")
            eval_and_log(concept, "ti", gen_ti_dir)

        for variant in variants:
            # train_tag marks a *training* ablation, so it renames the weights too;
            # gen_suffix marks an *inference-only* ablation, so it deliberately does not --
            # that's what lets a different --best_of_n reuse the same trained weights.
            label = f"{variant}{args.train_tag}{args.gen_suffix}"
            db_dir = f"db_{variant}_{concept}{args.train_tag}"
            db_weights = os.path.join(db_dir, "pytorch_lora_weights.safetensors")
            gen_db_dir = os.path.join("generated", f"{concept}_{label}")

            if os.path.exists(db_weights):
                print(f"  [skip] {label} already trained -> {db_weights}", flush=True)
            else:
                safe_run(
                    [PY, "train_dreambooth_lora.py",
                     "--data_dir", f"dataset/{concept}", "--output_dir", db_dir,
                     "--instance_prompt", f"a photo of {trigger}"] + common_flags
                    + variant_args(variant, concept, class_noun)
                    + shlex.split(args.extra_train_args),
                    log_path, f"{concept}: {label} train")

            if os.path.exists(db_weights) and not os.path.exists(os.path.join(gen_db_dir, "9.png")):
                safe_run(
                    [PY, "generate.py", "--method", "dreambooth", "--lora_dir", db_dir,
                     "--concept", concept, "--trigger_phrase", trigger, "--output_dir", gen_db_dir,
                     "--best_of_n", args.best_of_n],
                    log_path, f"{concept}: {label} generate")

            eval_and_log(concept, label, gen_db_dir)

    print("\n===== DONE =====", flush=True)
    if failures:
        print(f"{len(failures)} step(s) failed, see logs/*.log:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
    else:
        print("no failures.", flush=True)


if __name__ == "__main__":
    main()
