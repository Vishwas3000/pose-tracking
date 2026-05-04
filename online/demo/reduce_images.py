"""Resize a folder of images down to the bundle's reference resolution.

Why:
  - Smaller JPEGs decode faster (PIL → tensor takes a noticeable fraction of
    per-frame time on multi-MP photos).
  - At 1920×1440 the bundle's K applies directly — no scaling step downstream.
  - Smaller files are faster to ship/inspect.

Usage:
    python -m online.demo.reduce_images \\
        --in-dir  offline/data/test_images \\
        --out-dir offline/data/test_images_reduced \\
        --target-size 1920x1440
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",       required=True, type=Path)
    ap.add_argument("--out-dir",      required=True, type=Path)
    ap.add_argument("--target-size",  default="1920x1440",
                    help="Longer side fits within W x H; aspect preserved.")
    ap.add_argument("--quality",      type=int, default=85,
                    help="JPEG quality 1-100 (default 85)")
    ap.add_argument("--strip-exif-rotation", action="store_true",
                    help="Bake EXIF orientation into pixel data and clear the tag. "
                         "Off by default — keeps raw sensor orientation to match "
                         "the bundle's session frames.")
    args = ap.parse_args()

    tw, th = (int(x) for x in args.target_size.lower().split("x"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {args.in_dir}")
        return

    print(f"Reducing {len(files)} images: target {tw}x{th}, jpeg quality {args.quality}")
    print(f"  EXIF rotation: {'baked into pixels' if args.strip_exif_rotation else 'preserved'}")

    for fp in files:
        with Image.open(fp) as im:
            orig_w, orig_h = im.size
            if args.strip_exif_rotation:
                im = ImageOps.exif_transpose(im)
                orig_w, orig_h = im.size
            scale = min(tw / orig_w, th / orig_h)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            resized = im.resize((new_w, new_h), Image.BILINEAR)

            out_path = args.out_dir / fp.name
            save_kwargs = {"quality": args.quality, "optimize": True}
            # Preserve EXIF unless we baked it in
            if not args.strip_exif_rotation and "exif" in im.info:
                save_kwargs["exif"] = im.info["exif"]
            resized.save(out_path, **save_kwargs)

        in_kb  = fp.stat().st_size // 1024
        out_kb = out_path.stat().st_size // 1024
        print(f"  {fp.name:30s} {orig_w}x{orig_h} -> {new_w}x{new_h}  "
              f"{in_kb:>5d} KB -> {out_kb:>5d} KB  (-{100*(1 - out_kb/in_kb):.0f}%)")


if __name__ == "__main__":
    main()
