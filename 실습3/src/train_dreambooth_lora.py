"""DreamBooth + LoRA fine-tuning of SD3.5-medium, adapted from Day2_dreambooth.ipynb
for local low-VRAM (6GB) GPUs: the frozen transformer is loaded in 4-bit (nf4) via
bitsandbytes so only the small LoRA adapter needs full-precision gradients/optimizer state.

    python train_dreambooth_lora.py --data_dir dataset/pet_cat5 --output_dir db_lora_pet_cat5 \
        --instance_prompt "a photo of sks cat" --max_train_steps 5   # smoke test first
"""

import argparse
import math
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import BitsAndBytesConfig, StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# This script never uses torchao-quantized layers (only plain nn.Linear targets,
# with bitsandbytes handling 4-bit for the frozen base model), but peft's LoRA
# dispatcher still probes torchao's version on every adapter injection and raises
# ImportError on environments with an old torchao build (e.g. Colab's preinstalled
# 0.10.0). Short-circuit that probe so it never reaches the version check.
import peft.tuners.lora.torchao as _peft_lora_torchao
_peft_lora_torchao.dispatch_torchao = lambda *a, **k: None


class DreamBoothDataset(Dataset):
    def __init__(self, data_root, prompt, size=256, center_crop=False, no_flip=False):
        self.size = size
        self.prompt = prompt
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Instance images root {self.data_root} doesn't exist.")
        self.image_paths = [p for p in self.data_root.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        # Resize() scales the *shorter* side, so non-square references keep extra length
        # on the long axis that the crop then discards. With only 6-9 instance images a
        # random crop can clip the subject out of frame (actionfigure_2's references are
        # all ~0.8 aspect, so every one of them loses ~54px somewhere), and a horizontal
        # flip teaches a mirrored identity for asymmetric subjects. Both are on by
        # default to match the original notebook; these flags exist to ablate them.
        crop = transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size)
        ops = [
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
        ]
        if not no_flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(ops)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return {"instance_images": self.image_transforms(image), "instance_prompt": self.prompt}


def collate_fn(examples):
    pixel_values = torch.stack([e["instance_images"] for e in examples])
    return {"pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float()}


def sd3_flow_matching_loss(transformer, vae, noise_scheduler, pixel_values, prompt_embeds, pooled_prompt_embeds,
                            weight_dtype, vae_shift_factor, vae_scale_factor):
    with torch.no_grad():
        latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
        model_input = ((latents - vae_shift_factor) * vae_scale_factor).to(dtype=weight_dtype)

    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps.to(model_input.device)[indices]
    sigmas = noise_scheduler.sigmas.to(device=model_input.device, dtype=weight_dtype)[indices].flatten().reshape(bsz, 1, 1, 1)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    model_pred = transformer(
        hidden_states=noisy_model_input, timestep=timesteps,
        encoder_hidden_states=prompt_embeds.expand(bsz, -1, -1),
        pooled_projections=pooled_prompt_embeds.expand(bsz, -1)).sample
    model_pred = model_pred * (-sigmas) + noisy_model_input
    weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
    return torch.mean((weighting * (model_pred.float() - model_input.float()) ** 2).reshape(bsz, -1), 1).mean()


def encode_prompt(text_encoders, tokenizers, prompt, device, joint_attention_dim=4096):
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds_list.append(prompt_embeds[0])
        prompt_embeds_list.append(prompt_embeds.hidden_states[-2])

    clip_prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    t5_prompt_embeds = torch.zeros(
        (clip_prompt_embeds.shape[0], 77, joint_attention_dim), device=device, dtype=clip_prompt_embeds.dtype)
    clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, joint_attention_dim - clip_prompt_embeds.shape[-1]))
    final_prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    final_pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)
    return final_prompt_embeds, final_pooled_prompt_embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--instance_prompt", default="a photo of sks cat")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--max_train_steps", type=int, default=300)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--use_dora", action="store_true", help="use DoRA instead of vanilla LoRA (peft use_dora=True)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", default="fp16", choices=["fp16", "no"])
    ap.add_argument("--no_4bit", action="store_true", help="disable nf4 quantization of the frozen transformer")
    ap.add_argument("--with_prior_preservation", action="store_true",
                     help="DreamBooth prior-preservation loss (Ruiz et al.) against generated class images")
    ap.add_argument("--class_data_dir", help="required for --with_prior_preservation")
    ap.add_argument("--class_prompt", help="required for --with_prior_preservation, e.g. 'a photo of a cat'")
    ap.add_argument("--num_class_images", type=int, default=100)
    ap.add_argument("--prior_loss_weight", type=float, default=1.0)
    ap.add_argument("--class_gen_steps", type=int, default=20, help="inference steps for generating class images")
    ap.add_argument("--center_crop", action="store_true",
                     help="center-crop instead of random-crop, so a non-square reference can't have "
                          "the subject clipped out of frame")
    ap.add_argument("--no_flip", action="store_true",
                     help="disable random horizontal flip, which otherwise teaches a mirrored "
                          "identity for asymmetric subjects (faces, logos, held objects)")
    args = ap.parse_args()

    if args.with_prior_preservation and not (args.class_data_dir and args.class_prompt):
        ap.error("--with_prior_preservation requires --class_data_dir and --class_prompt")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    print("Loading transformer" + (" (nf4 4-bit, frozen)" if not args.no_4bit else " (fp16)") + "...")
    quant_kwargs = {}
    if not args.no_4bit:
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=weight_dtype)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=weight_dtype, **quant_kwargs)
    transformer.enable_gradient_checkpointing()

    print("Loading pipeline (reusing the transformer above)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, transformer=transformer, text_encoder_3=None, tokenizer_3=None, torch_dtype=weight_dtype)

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    vae, noise_scheduler = pipe.vae, pipe.scheduler

    vae.requires_grad_(False)
    for te in text_encoders:
        te.requires_grad_(False)
    transformer.requires_grad_(False)

    vae.to(accelerator.device, dtype=torch.float32)
    for te in text_encoders:
        te.to(accelerator.device, dtype=weight_dtype)
    if args.no_4bit:
        transformer.to(accelerator.device, dtype=weight_dtype)

    if args.with_prior_preservation:
        os.makedirs(args.class_data_dir, exist_ok=True)
        existing = [p for p in Path(args.class_data_dir).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        n_needed = args.num_class_images - len(existing)
        if n_needed > 0:
            print(f"Generating {n_needed} class images for prior preservation ({args.class_prompt!r})...")
            transformer.eval()
            gen = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            with torch.no_grad():
                for i in tqdm(range(n_needed), desc="Class images"):
                    # request raw latents and decode manually: the pipeline's own decode step
                    # doesn't cast latents to vae.dtype, which crashes since vae is forced to
                    # float32 above (line 168) while the transformer runs in weight_dtype.
                    latents = pipe(prompt=args.class_prompt, prompt_2=args.class_prompt,
                                    num_inference_steps=args.class_gen_steps, guidance_scale=7.0,
                                    height=args.resolution, width=args.resolution, generator=gen,
                                    output_type="latent").images
                    latents = latents.to(vae.dtype)
                    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                    decoded = vae.decode(latents, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
                    image.save(os.path.join(args.class_data_dir, f"class_{len(existing) + i}.png"))
            transformer.train()
            # pipe(...) calls scheduler.set_timesteps(class_gen_steps, ...) internally, which
            # shrinks noise_scheduler.timesteps/sigmas (same object, aliased above) to only
            # class_gen_steps entries. Restore the full training schedule before the loss
            # function indexes into it with indices in [0, num_train_timesteps).
            noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=accelerator.device)

    transformer.add_adapter(LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, use_dora=args.use_dora,
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]))
    if args.mixed_precision == "fp16":
        cast_training_params(transformer, dtype=torch.float32)

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    n_trainable = sum(p.numel() for p in params_to_optimize)
    print(f"Trainable LoRA params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps)

    train_dataset = DreamBoothDataset(args.data_dir, args.instance_prompt, size=args.resolution,
                                       center_crop=args.center_crop, no_flip=args.no_flip)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"{len(train_dataset)} instance images from {args.data_dir}")

    to_prepare = [transformer, optimizer, train_dataloader, lr_scheduler]
    class_dataloader = None
    if args.with_prior_preservation:
        class_dataset = DreamBoothDataset(args.class_data_dir, args.class_prompt, size=args.resolution)
        class_dataloader = torch.utils.data.DataLoader(
            class_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        to_prepare.append(class_dataloader)

    prepared = accelerator.prepare(*to_prepare)
    if args.with_prior_preservation:
        transformer, optimizer, train_dataloader, lr_scheduler, class_dataloader = prepared
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = prepared

    print("Pre-calculating text embeddings...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, args.instance_prompt, accelerator.device, transformer.config.joint_attention_dim)
        class_prompt_embeds = class_pooled_prompt_embeds = None
        if args.with_prior_preservation:
            class_prompt_embeds, class_pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, args.class_prompt, accelerator.device, transformer.config.joint_attention_dim)
    del text_encoders, tokenizers, pipe
    free_memory()

    from itertools import cycle
    class_dataloader_iter = cycle(class_dataloader) if class_dataloader is not None else None

    vae_shift_factor, vae_scale_factor = vae.config.shift_factor, vae.config.scaling_factor
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f"Steps/epoch={num_update_steps_per_epoch}, epochs={num_train_epochs}, total steps={args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    try:
        for epoch in range(num_train_epochs):
            transformer.train()
            for batch in train_dataloader:
                with accelerator.accumulate(transformer):
                    loss = sd3_flow_matching_loss(
                        transformer, vae, noise_scheduler, batch["pixel_values"],
                        prompt_embeds, pooled_prompt_embeds, weight_dtype, vae_shift_factor, vae_scale_factor)

                    if class_dataloader_iter is not None:
                        class_batch = next(class_dataloader_iter)
                        prior_loss = sd3_flow_matching_loss(
                            transformer, vae, noise_scheduler, class_batch["pixel_values"],
                            class_prompt_embeds, class_pooled_prompt_embeds, weight_dtype, vae_shift_factor, vae_scale_factor)
                        loss = loss + args.prior_loss_weight * prior_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    progress_bar.set_postfix(loss=loss.detach().item())
                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break
    except torch.cuda.OutOfMemoryError:
        print("\n=== CUDA OOM ===")
        print(torch.cuda.memory_summary(abbreviated=True))
        print("Try: --resolution 128, or --train_batch_size 1 (already default), or --no_4bit off (keep 4bit on).")
        raise

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir, transformer_lora_layers=get_peft_model_state_dict(unwrapped_transformer))
        print(f"Saved LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    main()
