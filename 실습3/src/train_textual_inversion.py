"""Textual Inversion fine-tuning of SD3.5-medium, adapted from Day2_textual_inversion.ipynb
for local low-VRAM (6GB) GPUs: the frozen transformer is loaded in 4-bit (nf4) via
bitsandbytes. Only the two placeholder-token embedding rows (CLIP-L, CLIP-G) are trained,
so they stay full precision while everything else stays frozen/quantized.

    python train_textual_inversion.py --data_dir dataset/pet_cat5 --output_dir ti_pet_cat5 \
        --placeholder_token "<pet_cat5>" --initializer_token cat --max_train_steps 5   # smoke test first
"""

import argparse
import glob
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.training_utils import free_memory


class TextualInversionDataset(Dataset):
    def __init__(self, data_root, size, placeholder_prompt):
        self.image_paths = sorted(
            glob.glob(os.path.join(data_root, "*.jpg")) + glob.glob(os.path.join(data_root, "*.jpeg")) +
            glob.glob(os.path.join(data_root, "*.png")))
        self.placeholder_prompt = placeholder_prompt
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {"pixel_values": self.image_transforms(image), "prompt": self.placeholder_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--placeholder_token", default="<pet_cat5>")
    ap.add_argument("--initializer_token", default="cat")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=300)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    print("Loading transformer" + (" (nf4 4-bit, frozen)" if not args.no_4bit else " (fp16)") + "...")
    quant_kwargs = {}
    if not args.no_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch_dtype, **quant_kwargs)
    transformer.enable_gradient_checkpointing()
    if args.no_4bit:
        transformer.to(device)

    print("Loading pipeline (reusing the transformer above)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, transformer=transformer, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch_dtype)

    tokenizer_one, tokenizer_two = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_one, text_encoder_two = pipe.text_encoder, pipe.text_encoder_2
    vae, scheduler = pipe.vae, pipe.scheduler

    tokenizers_list = [tokenizer_one, tokenizer_two]
    text_encoders_list = [text_encoder_one, text_encoder_two]
    for tokenizer in tokenizers_list:
        tokenizer.add_tokens(args.placeholder_token)

    placeholder_token_id_one = tokenizer_one.convert_tokens_to_ids(args.placeholder_token)
    initializer_token_id_one = tokenizer_one.convert_tokens_to_ids(args.initializer_token)
    placeholder_token_id_two = tokenizer_two.convert_tokens_to_ids(args.placeholder_token)
    initializer_token_id_two = tokenizer_two.convert_tokens_to_ids(args.initializer_token)

    for encoder, tokenizer, p_id, i_id in [
        (text_encoder_one, tokenizer_one, placeholder_token_id_one, initializer_token_id_one),
        (text_encoder_two, tokenizer_two, placeholder_token_id_two, initializer_token_id_two),
    ]:
        encoder.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            token_embeds = encoder.get_input_embeddings().weight.data
            token_embeds[p_id] = token_embeds[i_id].clone()

    for encoder, tokenizer in zip(text_encoders_list, tokenizers_list):
        encoder.text_model.eos_token_id = tokenizer.eos_token_id

    vae.requires_grad_(False)
    for encoder in text_encoders_list:
        encoder.requires_grad_(False)
        encoder.get_input_embeddings().requires_grad_(True)

    from itertools import chain
    params_to_train = chain(
        text_encoder_one.get_input_embeddings().parameters(),
        text_encoder_two.get_input_embeddings().parameters(),
    )

    vae.to(device)
    text_encoder_one.to(device, dtype=torch.float32)
    text_encoder_two.to(device, dtype=torch.float32)

    train_prompt = f"a photo of a {args.placeholder_token}"
    train_dataset = TextualInversionDataset(args.data_dir, args.resolution, train_prompt)
    print(f"{len(train_dataset)} images from {args.data_dir}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate)

    steps_per_epoch = len(train_dataloader)
    num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)
    print(f"{steps_per_epoch} steps/epoch, {num_train_epochs} epochs, {args.max_train_steps} steps total")

    orig_embeds = [enc.get_input_embeddings().weight.data.clone() for enc in text_encoders_list]
    index_no_updates = []
    for enc, p_id in zip(text_encoders_list, [placeholder_token_id_one, placeholder_token_id_two]):
        keep = torch.ones(enc.get_input_embeddings().weight.shape[0], dtype=torch.bool)
        keep[p_id] = False
        index_no_updates.append(keep)

    text_encoder_one.train()
    text_encoder_two.train()
    progress_bar = tqdm(total=args.max_train_steps, desc="Training")
    global_step = 0

    try:
        for epoch in range(num_train_epochs):
            for batch in train_dataloader:
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device, dtype=torch_dtype)).latent_dist.sample()
                    clean_latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                noise = torch.randn_like(clean_latents)
                bsz = clean_latents.shape[0]
                sigmas = torch.rand(bsz, device=clean_latents.device, dtype=torch_dtype)
                s_view = sigmas.view(-1, 1, 1, 1)
                noisy_latents = (1 - s_view) * clean_latents + s_view * noise
                target = noise - clean_latents
                timesteps = sigmas.float() * scheduler.config.num_train_timesteps

                prompt_embeds_output, _, pooled_prompt_embeds_output, _ = pipe.encode_prompt(
                    prompt=batch["prompt"], prompt_2=batch["prompt"], prompt_3=None,
                    device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)

                prompt_embeds = prompt_embeds_output.to(dtype=torch_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds_output.to(dtype=torch_dtype)

                model_pred = transformer(
                    hidden_states=noisy_latents, timestep=timesteps,
                    encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    for enc, orig, keep in zip(text_encoders_list, orig_embeds, index_no_updates):
                        enc.get_input_embeddings().weight[keep] = orig[keep]

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(epoch=f"{epoch + 1}/{num_train_epochs}", loss=f"{loss.item():.4f}")
                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break
    except torch.cuda.OutOfMemoryError:
        print("\n=== CUDA OOM ===")
        print(torch.cuda.memory_summary(abbreviated=True))
        print("Try: --resolution 128, or --train_batch_size 1 (already default).")
        raise

    progress_bar.close()
    os.makedirs(args.output_dir, exist_ok=True)
    learned_embed_one = text_encoder_one.get_input_embeddings().weight[placeholder_token_id_one].detach().cpu()
    learned_embed_two = text_encoder_two.get_input_embeddings().weight[placeholder_token_id_two].detach().cpu()
    torch.save(
        {"placeholder_token": args.placeholder_token, "clip_l": learned_embed_one, "clip_g": learned_embed_two},
        os.path.join(args.output_dir, "learned_embeds.bin"))
    print(f"Saved learned embeddings to {args.output_dir}/learned_embeds.bin")


if __name__ == "__main__":
    main()
