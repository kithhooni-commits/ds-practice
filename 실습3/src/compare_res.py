"""Side-by-side 256px-trained vs 512px-trained generations, at native size plus a zoom.

The HTML viewer scales thumbnails to ~280px, which throws away exactly the detail this
comparison is about. So this writes full-resolution pairs and a 2x centre crop next to
them — the crop is where a training-resolution difference actually shows up.

    python compare_res.py --a v2_review/generated --b r512_review/r512 \
        --concepts actionfigure_2,person_3,scene_waterfall
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

CLASS_PROMPT = {
    "actionfigure_2": "action figure",
    "person_3": "person",
    "scene_waterfall": "waterfall",
    "decoritems_woodenpot": "wooden pot",
    "furniture_sofa2": "sofa",
    "instrument_music2": "guitar",
    "luggage_backpack1": "backpack",
    "pet_cat5": "cat",
    "transport_tank": "tank",
    "wearable_jacket1": "jacket",
}

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arial.ttf"]


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(root, concept, method, i):
    p = os.path.join(root, f"{concept}_{method}", f"{i}.png")
    return Image.open(p).convert("RGB") if os.path.exists(p) else None


def zoom(im, factor=2):
    """Centre crop at 1/factor of the frame, scaled back up — nearest-neighbour so the
    comparison shows the pixels that exist rather than the resampler's guesses."""
    w, h = im.size
    s = min(w, h) // factor
    l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((w, h), Image.NEAREST)


def build(concept, a_root, b_root, a_method, b_method, prompts, out, tile=380,
          labels=("256px 학습", "512px 학습"), title="학습 해상도 비교"):
    pad, head, cap = 12, 58, 30
    cols = [labels[0], labels[1], f"{labels[0]} 확대 2x", f"{labels[1]} 확대 2x"]
    n = len(prompts)
    W = pad + 4 * (tile + pad)
    H = head + n * (tile + cap + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f_head, f_cap, f_col = font(21), font(13), font(15)

    d.text((pad, 12), f"{concept}  ({CLASS_PROMPT[concept]})  —  {title}",
           fill="black", font=f_head)
    for c, name in enumerate(cols):
        d.text((pad + c * (tile + pad), head - 20), name, fill="#444444", font=f_col)

    for r, i in enumerate(prompts):
        a, b = load(a_root, concept, a_method, i), load(b_root, concept, b_method, i)
        y = head + r * (tile + cap + pad)
        for c, im in enumerate([a, b, zoom(a) if a else None, zoom(b) if b else None]):
            x = pad + c * (tile + pad)
            if im is None:
                d.rectangle([x, y, x + tile, y + tile], fill="#eeeeee")
                d.text((x + 8, y + tile // 2), "(missing)", fill="#999999", font=f_cap)
                continue
            sheet.paste(im.resize((tile, tile), Image.LANCZOS), (x, y))
            d.rectangle([x, y, x + tile, y + tile], outline="#cccccc")
        d.text((pad, y + tile + 7), f"프롬프트 {i}", fill="#666666", font=f_cap)

    sheet.save(out, quality=94)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="v2_review/generated", help="256px-trained run")
    ap.add_argument("--b", default="r512_review/r512", help="512px-trained run")
    ap.add_argument("--a_method", default="lora")
    ap.add_argument("--b_method", default="lora_r512")
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--prompts", default="0,1,5", help="which prompt indices to show")
    ap.add_argument("--out_dir", default="compare")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = [int(x) for x in args.prompts.split(",")]
    for concept in args.concepts.split(","):
        out = os.path.join(args.out_dir, f"res_{concept}.jpg")
        build(concept, args.a, args.b, args.a_method, args.b_method, prompts, out)
        print(f"  {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
