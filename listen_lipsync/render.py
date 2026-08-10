#!/usr/bin/env python3
"""把口型时间轴渲染成视频：静态立绘 + 按 cue 绘制嘴型 + 节拍身体浮动。

嘴部位置来自 lbpcascade_animeface 检出的人脸框 + 动漫脸比例推算（见 detect_mouth.py），
可用 --mouth x,y,w,h 覆盖。嘴型这里用程序化绘制（椭圆/弧线）而非手绘 sprite —— 目的是
验证时间轴，真实产出应替换成画好的 9 张嘴部贴图。

用法（需要 PIL + numpy，GPT-SoVITS venv 即可）：
  python render.py --img figure.png --cues cues.json --audio clip.wav --out out.mp4 \
      --mouth 596,583,209,146
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

# 每个 Rhubarb 口型的 (张开度 0-1, 横向宽度比 0-1)。
# A 闭嘴, B 微开, C 中开, D 大开, E 圆略开, F 收圆, G 唇齿, H=L, X 静音闭合
SHAPE = {
    "A": (0.00, 0.62), "B": (0.16, 0.60), "C": (0.42, 0.72), "D": (0.85, 0.78),
    "E": (0.40, 0.46), "F": (0.30, 0.34), "G": (0.14, 0.52), "H": (0.34, 0.56),
    "X": (0.00, 0.58),
}


def draw_mouth(base, box, shape, tint):
    """在 base(RGBA) 上按 shape 画嘴。box=(x,y,w,h) 为嘴部 ROI。"""
    x, y, w, h = box
    op, wd = SHAPE.get(shape, SHAPE["B"])
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cx, cy = x + w / 2, y + h / 2
    mw = w * wd
    mh = max(2.0, h * 0.92 * op)
    dark, lip = (58, 28, 36, 255), tint
    if op < 0.06:
        # 闭嘴：只画一条唇线
        d.line([(cx - mw / 2, cy), (cx + mw / 2, cy)], fill=(120, 70, 78, 235),
               width=max(2, int(h * 0.06)))
    else:
        d.ellipse([cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2], fill=dark)
        # 下唇高光/舌，让张口有层次
        if op > 0.35:
            d.ellipse([cx - mw * 0.30, cy + mh * 0.06, cx + mw * 0.30, cy + mh * 0.46],
                      fill=(176, 84, 96, 220))
        d.arc([cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2], 200, 340,
              fill=lip, width=max(2, int(h * 0.05)))
    return Image.alpha_composite(base, lay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mouth", required=True, help="x,y,w,h 嘴部 ROI")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, help="只渲染前 N 秒")
    ap.add_argument("--size", type=int, default=720, help="输出边长")
    ap.add_argument("--bob", type=float, default=5.0, help="身体上下浮动像素")
    args = ap.parse_args()

    box = [int(v) for v in args.mouth.split(",")]
    cues = json.load(open(args.cues))
    mc = cues["mouthCues"]
    dur = cues["metadata"]["duration"]
    if args.seconds:
        dur = min(dur, args.seconds)
    n = int(dur * args.fps)

    src = Image.open(args.img).convert("RGBA")
    # 预生成每种口型一张整图，避免逐帧重画
    tint = (196, 118, 128, 235)
    cache = {}
    for s in SHAPE:
        im = draw_mouth(src, box, s, tint)
        cache[s] = im

    def shape_at(t):
        for c in mc:
            if c["start"] <= t < c["end"]:
                return c["value"]
        return "X"

    tmp = tempfile.mkdtemp(prefix="lipsync_")
    W = args.size
    for i in range(n):
        t = i / args.fps
        s = shape_at(t)
        im = cache[s]
        # 呼吸/律动：整体上下浮动（简单正弦，真实版本可接 librosa 节拍）
        off = int(round(args.bob * np.sin(2 * np.pi * t * 0.9)))
        canvas = Image.new("RGBA", src.size, (0, 0, 0, 255))
        canvas.paste(im, (0, off), im)
        canvas.convert("RGB").resize((W, W), Image.LANCZOS).save(
            os.path.join(tmp, f"{i:06d}.png"), compress_level=1)
        if i % 50 == 0:
            print(f"\r帧 {i}/{n}", end="", flush=True)
    print(f"\r帧 {n}/{n} 完成，编码中...")

    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(tmp, "%06d.png"), "-i", args.audio,
           "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-c:a", "aac", "-b:a", "160k", "-shortest", args.out]
    subprocess.run(cmd, check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print(f"写出 {args.out}  ({n} 帧 / {dur:.1f}s / {args.fps}fps)")


if __name__ == "__main__":
    main()
