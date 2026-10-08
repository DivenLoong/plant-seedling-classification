"""Draw the method pipeline figure used in the paper."""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

HEI = r"C:\Windows\Fonts\simhei.ttf"
SUN = r"C:\Windows\Fonts\simsun.ttc"
W, H = 1800, 560
BORDER = (40, 40, 40)
BOX_FILL = (255, 255, 255)
ACCENT = (31, 56, 100)


def font(path, size):
    return ImageFont.truetype(path, size)


def box(draw, x, y, w, h, title, detail, accent=False):
    draw.rectangle([x, y, x + w, y + h], fill=BOX_FILL,
                   outline=ACCENT if accent else BORDER, width=3)
    f_title = font(HEI, 26)
    f_detail = font(SUN, 20)
    if detail:
        draw.text((x + w / 2, y + h * 0.34), title, font=f_title,
                  fill=(0, 0, 0), anchor="mm")
        draw.text((x + w / 2, y + h * 0.70), detail, font=f_detail,
                  fill=(60, 60, 60), anchor="mm")
    else:
        draw.text((x + w / 2, y + h / 2), title, font=f_title,
                  fill=(0, 0, 0), anchor="mm")


def arrow(draw, x1, y1, x2, y2, dashed=False):
    if dashed:
        # simple dashed straight line, horizontal or vertical only
        step = 18
        if y1 == y2:
            xs = range(min(x1, x2), max(x1, x2), step * 2)
            for x in xs:
                draw.line([x, y1, min(x + step, max(x1, x2)), y1],
                          fill=BORDER, width=3)
        else:
            ys = range(min(y1, y2), max(y1, y2), step * 2)
            for y in ys:
                draw.line([x1, y, x1, min(y + step, max(y1, y2))],
                          fill=BORDER, width=3)
        return
    draw.line([x1, y1, x2, y2], fill=BORDER, width=3)
    if y1 == y2:  # horizontal head
        d = 1 if x2 > x1 else -1
        draw.polygon([(x2, y2), (x2 - 16 * d, y2 - 9), (x2 - 16 * d, y2 + 9)],
                     fill=BORDER)
    else:  # vertical head
        d = 1 if y2 > y1 else -1
        draw.polygon([(x2, y2), (x2 - 9, y2 - 16 * d), (x2 + 9, y2 - 16 * d)],
                     fill=BORDER)


def line(draw, x1, y1, x2, y2):
    """Plain connector segment, used for the elbows of a polyline route."""
    draw.line([x1, y1, x2, y2], fill=BORDER, width=3)


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    bw, bh = 400, 120
    y1, y2 = 60, 330
    xs = [40, 480, 920, 1360]

    box(d, xs[0], y1, bw, bh, "输入图像", "49×49 至 2840×2132")
    box(d, xs[1], y1, bw, bh, "前景裁剪与尺度归一化", "HSV 掩膜 + 包围盒 + 补正方形")
    box(d, xs[2], y1, bw, bh, "预训练骨干两阶段微调", "ConvNeXt / EfficientNet")
    box(d, xs[3], y1, bw, bh, "多视图 TTA 与五折集成", "2 尺度 × 4 旋转 × 2 镜像")

    for i in range(3):
        arrow(d, xs[i] + bw, y1 + bh / 2, xs[i + 1], y1 + bh / 2)

    # elbow connector from the end of row 1 back to the start of row 2
    ex = xs[3] + bw / 2
    line(d, ex, y1 + bh, ex, y2 - 30)
    line(d, ex, y2 - 30, xs[0] + 220, y2 - 30)
    arrow(d, xs[0] + 220, y2 - 30, xs[0] + 220, y2)

    box(d, xs[0], y2, 440, bh, "交叉验证 OOF 混淆矩阵", "非负最小二乘反演先验", True)
    box(d, 560, y2, 440, bh, "类别先验校正", "argmax πc · p(c|x)", True)
    box(d, 1080, y2, 400, bh, "预测输出", "378 张测试图类别")
    arrow(d, xs[0] + 440, y2 + bh / 2, 560, y2 + bh / 2)
    arrow(d, 1000, y2 + bh / 2, 1080, y2 + bh / 2)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_pipeline.png")
    img.save(out, dpi=(300, 300))
    print("written:", out, img.size)


if __name__ == "__main__":
    main()
