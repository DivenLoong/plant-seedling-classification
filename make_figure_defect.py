"""Figure: the crop/resize ordering defect, on a real training image."""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

HEI = r"C:\Windows\Fonts\simhei.ttf"
SUN = r"C:\Windows\Fonts\simsun.ttc"
CELL = 430
GAP = 40


def main() -> None:
    path = os.path.join(P.TRAIN_DIR, "Loose Silky-bent", "ffqwnlenki.png")
    cfg = P.CropConfig()
    original = P.load_rgb(path)
    box = P.plant_box(original, cfg)

    # what the pipeline did: downscale first, then crop with original-image coordinates
    small = original.resize((448, 448), Image.BILINEAR)
    buggy = small.crop(box)
    # what it should do: crop first, then downscale
    fixed = original.crop(box).resize((448, 448), Image.BILINEAR)

    canvas = Image.new("RGB", (CELL * 3 + GAP * 4, CELL + 92), "white")
    draw = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(HEI, 25)
    f_note = ImageFont.truetype(SUN, 21)
    labels = [
        ("原图 2030×2030", "原始图像无黑色区域"),
        ("错误顺序：先缩放再裁剪", "越界区域被填成黑色，黑色像素 95.1%"),
        ("修复顺序：先裁剪再缩放", "黑色像素 0.0%"),
    ]
    for i, (im, (title, note)) in enumerate(zip((original, buggy, fixed), labels)):
        tile = im.convert("RGB").copy()
        tile.thumbnail((CELL, CELL), Image.BILINEAR)
        x = GAP + i * (CELL + GAP)
        draw.text((x, 16), title, font=f_title, fill=(0, 0, 0))
        canvas.paste(tile, (x + (CELL - tile.width) // 2, 56))
        draw.text((x, 56 + CELL + 6), note, font=f_note, fill=(70, 70, 70))

    out = os.path.join(P.ROOT, "figure_defect.png")
    canvas.save(out, dpi=(300, 300))
    print("written:", out, canvas.size)


if __name__ == "__main__":
    main()
