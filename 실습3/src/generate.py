"""Generate the 10 evaluation images for a concept from a fine-tuned SD3.5-medium model
(either a Textual Inversion embedding or a DreamBooth+LoRA adapter).

Uses `enable_model_cpu_offload()` (like Day1_diffusers_tutorial.ipynb) instead of 4-bit
quantization, since inference has no optimizer/gradient memory to worry about and CPU
offload keeps whichever component isn't currently running off the 6GB GPU.

    python generate.py --method dreambooth --lora_dir db_lora_pet_cat5 \
        --concept pet_cat5 --trigger_phrase "sks cat" --output_dir generated/pet_cat5_dreambooth

    python generate.py --method ti --embeds_path ti_pet_cat5/learned_embeds.bin \
        --concept pet_cat5 --output_dir generated/pet_cat5_ti
"""

import argparse
import os

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPProcessor

CLIP_ID = "openai/clip-vit-base-patch32"


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


def load_prompts(prompts_dir, concept):
    path = os.path.join(prompts_dir, f"{concept}.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dreambooth", "ti"])
    ap.add_argument("--concept", required=True)
    ap.add_argument("--prompts_dir", default="prompts")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--lora_dir", help="required for --method dreambooth")
    ap.add_argument("--embeds_path", help="required for --method ti (learned_embeds.bin)")
    ap.add_argument("--trigger_phrase", default="sks cat", help="text substituted for {} (dreambooth only)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--num_inference_steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--best_of_n", type=int, default=1,
                     help="generate N candidates per prompt (different seeds) and keep the CLIP-highest-scoring one")
    args = ap.parse_args()

    if args.method == "dreambooth" and not args.lora_dir:
        ap.error("--lora_dir is required for --method dreambooth")
    if args.method == "ti" and not args.embeds_path:
        ap.error("--embeds_path is required for --method ti")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    print("Loading pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch_dtype)

    if args.method == "dreambooth":
        pipe.load_lora_weights(args.lora_dir)
        pipe.enable_lora()
        subject = args.trigger_phrase
        prompt_2_subject = subject
    else:
        state_dict = torch.load(args.embeds_path)
        placeholder_token = state_dict["placeholder_token"]
        learned_embeds = [state_dict["clip_l"], state_dict["clip_g"]]
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        for te, tok, embed in zip(text_encoders, tokenizers, learned_embeds):
            tok.add_tokens(placeholder_token)
            te.resize_token_embeddings(len(tok))
            token_id = tok.convert_tokens_to_ids(placeholder_token)
            with torch.no_grad():
                te.get_input_embeddings().weight.data[token_id] = embed.to(te.device, dtype=te.dtype)
        for te, tok in zip(text_encoders, tokenizers):
            te.text_model.eos_token_id = tok.eos_token_id
        subject = placeholder_token
        prompt_2_subject = placeholder_token

    pipe.enable_model_cpu_offload()

    clip_model = clip_processor = None
    if args.best_of_n > 1:
        print(f"Best-of-{args.best_of_n}: loading CLIP scorer ({CLIP_ID})...")
        clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

    prompts = load_prompts(args.prompts_dir, args.concept)
    print(f"{len(prompts)} prompts loaded for concept '{args.concept}'")

    os.makedirs(args.output_dir, exist_ok=True)
    for i, template in enumerate(prompts):
        prompt = template.replace("{}", subject)
        prompt_2 = template.replace("{}", prompt_2_subject)

        candidates = []
        for n in range(args.best_of_n):
            generator = torch.Generator(device=device).manual_seed(args.seed + i * 1000 + n)
            image = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            ).images[0]
            candidates.append(image)

        if len(candidates) == 1:
            best_image, best_idx = candidates[0], 0
        else:
            with torch.no_grad():
                inputs = clip_processor(text=[prompt], images=candidates, return_tensors="pt", padding=True).to(device)
                text_out = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                img_out = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                text_feat = clip_embed(text_out, clip_model.text_projection)
                img_feat = clip_embed(img_out, clip_model.visual_projection)
                sims = F.cosine_similarity(img_feat, text_feat.expand(len(candidates), -1))
                best_idx = int(sims.argmax().item())
            best_image = candidates[best_idx]

        out_path = os.path.join(args.output_dir, f"{i}.png")
        best_image.save(out_path)
        suffix = f" (best of {len(candidates)}, picked #{best_idx})" if len(candidates) > 1 else ""
        print(f"[{i + 1}/{len(prompts)}] {prompt!r} -> {out_path}{suffix}")

    print(f"Done. Saved {len(prompts)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
