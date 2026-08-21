"""CLIP-based evaluation for the term project.
실행
----
    python evaluation.py --dataset ./dataset --prompts ./prompts --concept actionfigure_2 --images ./generated/actionfigure_2

    --dataset  레퍼런스 이미지의 상위 폴더 (컨셉 폴더들이 그 안에 있음)
    --prompts  프롬프트 txt 들이 있는 폴더
    --images   해당 컨셉의 생성 이미지 10장이 될 폴더 (컨셉 하나만 지정)

i번째 생성 이미지는 i번째 프롬프트로 만든 것으로 간주하여 짝지어 채점합니다.
파일명 정렬 순서가 프롬프트 순서와 같아야 합니다 (예: 0.png ... 9.png).
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"          # ProjectOverview.pdf 지정 모델 (CLIP-B/32)

CLASS_PROMPT = {                                  # CustomConcept101 dataset.json 의 class_prompt
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


def clip_embed(out, projection):
    """Normalize what CLIPModel.get_*_features() returned into a [N, proj_dim] tensor.

    Across transformers versions this returns either the final projected embedding or
    the raw sub-model output, and even within one version the wrapped value is
    sometimes already projected. So decide by shape, not by type: unwrap to a 2D
    tensor, then apply the projection only if it isn't already at proj_dim.
    """
    if not isinstance(out, torch.Tensor):
        out = getattr(out, "pooler_output", None)
        if out is None:
            raise RuntimeError("CLIP features had no pooler_output to fall back on")
    if out.ndim == 3:                      # [N, seq, hidden] -> take the pooled position
        out = out[:, 0]
    if out.shape[-1] != projection.out_features:
        out = projection(out)
    return out


ap = argparse.ArgumentParser(description="CLIP-B/32 evaluation (Text-to-Image / Image-to-Image)")
ap.add_argument("--dataset", required=True, help="레퍼런스 이미지 최상위 폴더 (dataset.zip 해제 위치)")
ap.add_argument("--prompts", required=True, help="프롬프트 txt 폴더 (prompt.zip 해제 위치)")
ap.add_argument("--concept", required=True, choices=sorted(CLASS_PROMPT), help="평가할 컨셉명")
ap.add_argument("--images", required=True, help="생성 이미지 10장이 들어있는 폴더")
args = ap.parse_args()

ref_paths = sorted(glob.glob(os.path.join(args.dataset, args.concept, "*")))
gen_paths = sorted(glob.glob(os.path.join(args.images, "*")))
prompts = [l.strip().replace("{}", CLASS_PROMPT[args.concept])
           for l in open(os.path.join(args.prompts, f"{args.concept}.txt"), encoding="utf-8") if l.strip()]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(CLIP_ID)

with torch.no_grad():
    reference = [Image.open(p).convert("RGB") for p in ref_paths]
    images = [Image.open(p).convert("RGB") for p in gen_paths]

    text_out = model.get_text_features(
        **processor(text=prompts, return_tensors="pt", padding=True).to(device))
    image_ref_out = model.get_image_features(
        **processor(images=reference, return_tensors="pt").to(device))
    image_gen_out = model.get_image_features(
        **processor(images=images, return_tensors="pt").to(device))

    text_embedding = clip_embed(text_out, model.text_projection)
    image_ref = clip_embed(image_ref_out, model.visual_projection)
    image_gen = clip_embed(image_gen_out, model.visual_projection)

n = min(len(image_gen), len(text_embedding))
t2i_sim = F.cosine_similarity(image_gen[:n], text_embedding[:n])                 # i번째끼리 짝
i2i_sim = F.cosine_similarity(image_gen.unsqueeze(1), image_ref.unsqueeze(0), dim=-1)  # 전체 쌍

print(f"Concept: {args.concept} ({CLASS_PROMPT[args.concept]})  |  reference {len(ref_paths)}장, generated {len(gen_paths)}장")
print(f"Text-to-Image Score: {t2i_sim.mean().item():.4f}")
print(f"Image-to-Image Score: {i2i_sim.mean().item():.4f}")
