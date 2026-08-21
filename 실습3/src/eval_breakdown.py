"""Per-prompt CLIP score breakdown across concepts and methods.

evaluation.py reports only the 10-prompt mean, which hides *which* prompts a method
fails. This scores every (concept, method, prompt) triple individually so a failure
like "prompt 9 (riding a dragon) collapses for every method" becomes a number instead
of an eyeball impression -- and so before/after comparisons can be read per prompt
rather than washed out in the mean.

Loads CLIP once and reuses it for every combination, which is why this is much faster
than looping evaluation.py.

    python eval_breakdown.py --concept actionfigure_2
    python eval_breakdown.py --all --generated baseline_review/generated --out baseline.csv
    python eval_breakdown.py --all --methods lora,lora_bo1        # ablation comparison

Writes a tidy CSV (one row per concept/method/prompt) plus a printed per-concept table.
"""

import argparse
import csv
import glob
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"
DINO_ID = "facebook/dino-vits16"   # the identity metric from the DreamBooth paper

CLASS_PROMPT = {  # kept in sync with evaluation.py's CLASS_PROMPT
    "actionfigure_2": "action figure",
    "decoritems_woodenpot": "wooden pot",
    "furniture_sofa2": "sofa",
    "instrument_music2": "guitar",
    "luggage_backpack1": "backpack",
    "person_3": "person",
    "pet_cat5": "cat",
    "scene_waterfall": "waterfall",
    "transport_tank": "tank",
    "wearable_jacket1": "jacket",
}

KNOWN_METHODS = ["ti", "lora", "dora", "lora_prior", "dora_prior"]


def clip_embed(out, projection):
    """Normalize what CLIPModel.get_*_features() returned into a [N, proj_dim] tensor.

    Across transformers versions this returns either the final projected embedding or
    the raw sub-model output, and even within one version the wrapped value is
    sometimes already projected. So decide by shape, not by type.
    """
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "pooler_output", None)
        if out is None:
            raise RuntimeError("CLIP features had no pooler_output to fall back on")
    if out.ndim == 3:
        out = out[:, 0]
    if out.shape[-1] != projection.out_features:
        out = projection(out)
    return out


def embed_images(model, processor, paths, device):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    out = model.get_image_features(**processor(images=imgs, return_tensors="pt").to(device))
    return clip_embed(out, model.visual_projection)


def embed_texts(model, processor, texts, device):
    out = model.get_text_features(
        **processor(text=texts, return_tensors="pt", padding=True).to(device))
    return clip_embed(out, model.text_projection)


def dino_embed(model, processor, paths, device):
    """DINO CLS-token embedding.

    CLIP image similarity turns out to track how much the *scene* matches the
    reference set, not whether the subject's identity survived -- a wide action shot
    scores low against close-up references even with the identity intact. DINO is
    self-supervised on images alone, so its features key on instance appearance
    rather than caption semantics, which is why the DreamBooth paper reports it
    alongside CLIP-I. Same convention here: CLS token, cosine similarity.
    """
    imgs = [Image.open(p).convert("RGB") for p in paths]
    out = model(**processor(images=imgs, return_tensors="pt").to(device))
    return out.last_hidden_state[:, 0]


def main():
    # some prompts carry non-ASCII (e.g. "cafe" spelled with an accent), which blows up
    # when Windows redirects stdout under a legacy codepage. Never let a print kill a run
    # that has minutes of GPU/CPU work behind it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", help="one concept, or a comma-separated list; omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--generated", default="generated")
    ap.add_argument("--methods", default=",".join(KNOWN_METHODS),
                    help="comma-separated method suffixes to look for under "
                         "<generated>/<concept>_<method>/")
    ap.add_argument("--out", default="results/breakdown.csv")
    ap.add_argument("--dino", action="store_true",
                    help=f"also score DINO image-to-image ({DINO_ID}). Roughly doubles "
                         "runtime but separates identity loss from framing mismatch, "
                         "which CLIP image similarity conflates")
    args = ap.parse_args()

    if not args.all and not args.concept:
        ap.error("pass --concept <name> or --all")

    concepts = sorted(CLASS_PROMPT) if args.all else [c for c in args.concept.split(",") if c]
    unknown = [c for c in concepts if c not in CLASS_PROMPT]
    if unknown:
        ap.error(f"unknown concept(s): {', '.join(unknown)}")
    methods = [m for m in args.methods.split(",") if m]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ({CLIP_ID}) on {device}...")
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    dino = dino_proc = None
    if args.dino:
        print(f"Loading DINO ({DINO_ID}) on {device}...")
        dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
        dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = 0
    csv_file = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["concept", "method", "prompt_idx", "prompt", "t2i", "i2i", "dino"])
    writer.writeheader()

    with torch.no_grad():
        for concept in concepts:
            noun = CLASS_PROMPT[concept]
            prompt_path = os.path.join(args.prompts, f"{concept}.txt")
            if not os.path.exists(prompt_path):
                print(f"[skip] {concept}: no {prompt_path}")
                continue
            with open(prompt_path, encoding="utf-8") as f:
                prompts = [l.strip().replace("{}", noun) for l in f if l.strip()]

            ref_paths = sorted(glob.glob(os.path.join(args.dataset, concept, "*")))
            if not ref_paths:
                print(f"[skip] {concept}: no reference images")
                continue

            # reference embeddings and prompt embeddings are shared across methods,
            # so compute them once per concept rather than once per method
            ref_emb = embed_images(model, processor, ref_paths, device)
            txt_emb = embed_texts(model, processor, prompts, device)
            ref_dino = dino_embed(dino, dino_proc, ref_paths, device) if dino else None

            present = {}
            for method in methods:
                gen_paths = sorted(glob.glob(os.path.join(
                    args.generated, f"{concept}_{method}", "*.png")))
                if not gen_paths:
                    continue
                gen_emb = embed_images(model, processor, gen_paths, device)
                n = min(len(gen_emb), len(txt_emb))
                t2i = F.cosine_similarity(gen_emb[:n], txt_emb[:n])
                # per generated image: mean similarity against every reference
                i2i = F.cosine_similarity(gen_emb.unsqueeze(1), ref_emb.unsqueeze(0), dim=-1).mean(dim=1)
                dn = None
                if dino:
                    gen_dino = dino_embed(dino, dino_proc, gen_paths, device)
                    dn = F.cosine_similarity(gen_dino.unsqueeze(1), ref_dino.unsqueeze(0), dim=-1).mean(dim=1)
                present[method] = (t2i, i2i, dn)
                for i in range(n):
                    writer.writerow({
                        "concept": concept, "method": method, "prompt_idx": i,
                        "prompt": prompts[i],
                        "t2i": round(t2i[i].item(), 4),
                        "i2i": round(i2i[i].item(), 4),
                        "dino": round(dn[i].item(), 4) if dn is not None else "",
                    })
                    n_rows += 1
            # flush per concept so a later crash can't discard earlier results
            csv_file.flush()

            if not present:
                print(f"[skip] {concept}: no generated images under {args.generated}")
                continue

            w = 30 if dino else 22
            print(f"\n===== {concept} ({noun}) =====")
            head = "  # " + "".join(f"{m:>{w}}" for m in present) + "   prompt"
            print(head)
            print("  " + "-" * (len(head) - 2))
            for i in range(len(prompts)):
                cells = ""
                for m, (t2i, i2i, dn) in present.items():
                    if i >= len(t2i):
                        cells += f"{'-':>{w}}"
                        continue
                    cell = f"{t2i[i].item():.4f}/{i2i[i].item():.4f}"
                    if dn is not None:
                        cell += f"/{dn[i].item():.4f}"
                    cells += f"{cell:>{w}}"
                print(f"  {i} {cells}   {prompts[i][:52]}")
            means = ""
            for t2i, i2i, dn in present.values():
                cell = f"{t2i.mean().item():.4f}/{i2i.mean().item():.4f}"
                if dn is not None:
                    cell += f"/{dn.mean().item():.4f}"
                means += f"{cell:>{w}}"
            label = "t2i/i2i/dino" if dino else "t2i/i2i"
            print(f"  ~ {means}   (mean, {label})")

    csv_file.close()
    if n_rows:
        print(f"\nWrote {n_rows} rows to {args.out}")
    else:
        print("\nNo scores produced -- check --generated / --methods.")


if __name__ == "__main__":
    main()
