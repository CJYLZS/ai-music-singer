#!/usr/bin/env python3
"""把 AI 生成的 3x3 口型九宫格切成 9 张 viseme sprite。

渲染端 (a25rig/index.html __rig.setVisemes) 需要每个口型一张透明 PNG；本脚本负责
切格、去背、裁到内容、统一尺寸。

九宫格的格位顺序（从左到右、从上到下）必须是：
    X(静音/闭)   A(闭唇 b/p/m)   B(i 展唇)
    C(e 中开)    D(a 大开)       E(o 圆中开)
    F(u 强圆唇)  G(f 唇齿)       H(ü 圆唇)

用法:
    python listen_lipsync/split_visemes.py grid.png -o data/pics/visemes
    # 白底/灰底会被自动去掉；若原图已带 alpha 则直接沿用
    python listen_lipsync/split_visemes.py grid.png -o out --rows 3 --cols 3
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

# 格位 → 口型字母（左到右、上到下）
ORDER = ["X", "A", "B", "C", "D", "E", "F", "G", "H"]
DESC = {
    "X": "静音/闭嘴", "A": "闭唇 b/p/m", "B": "i 展唇窄开", "C": "e 中开微展",
    "D": "a 大开", "E": "o 圆中开", "F": "u 强圆唇", "G": "f 唇齿", "H": "ü 圆唇高",
}


def kill_bg(im, tol):
    """把接近纯白/纯灰的背景变透明。已有有效 alpha 时原样返回。"""
    a = np.array(im.convert("RGBA"))
    if (a[:, :, 3] < 250).mean() > 0.02:      # 已经有像样的透明区，认为自带 alpha
        return Image.fromarray(a)
    rgb = a[:, :, :3].astype(int)
    # 背景色取四角众数，避免假设一定是白色
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                              rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    d = np.abs(rgb - bg).sum(axis=2)
    a[:, :, 3] = np.where(d < tol, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def trim(im):
    a = np.array(im)
    ys, xs = np.where(a[:, :, 3] > 20)
    if len(ys) == 0:
        return None
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def split_by_blobs(im, rows, cols, min_frac=0.0005):
    """按连通域切格，而不是按几何格线。

    生成图常带边框/格线，而且每个嘴在格内并不居中；按几何切会把边框一起切进去、
    或截掉张大的嘴。连通域直接给出每个嘴的真实 bbox，还能靠"bbox 接近整幅"识别并
    剔除边框域。返回按行列排序的 [(key, PIL.Image)]。
    """
    from scipy import ndimage
    a = np.array(im.convert("RGBA"))
    H, W = a.shape[:2]
    lab, n = ndimage.label(a[:, :, 3] > 20)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < H * W * min_frac:
            continue
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if w > W * 0.9 and h > H * 0.9:            # 边框/网格线
            continue
        blobs.append({"bbox": (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                      "cx": xs.mean(), "cy": ys.mean(), "lab": i})
    if len(blobs) != rows * cols:
        raise SystemExit(f"检测到 {len(blobs)} 个口型域，期望 {rows*cols} 个。"
                         f"请检查九宫格是否有空格或粘连。")
    blobs.sort(key=lambda b: b["cy"])
    out = []
    for r in range(rows):
        band = sorted(blobs[r * cols:(r + 1) * cols], key=lambda b: b["cx"])
        for b in band:
            x0, y0, x1, y1 = b["bbox"]
            cell = a[y0:y1, x0:x1].copy()
            # 只保留该连通域自身的像素，避免相邻格的边角混入
            m = lab[y0:y1, x0:x1] == b["lab"]
            cell[:, :, 3] = np.where(m, cell[:, :, 3], 0)
            out.append(Image.fromarray(cell))
    return list(zip(ORDER[: rows * cols], out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", help="3x3 九宫格图")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--tol", type=int, default=40, help="去背色差阈值")
    ap.add_argument("--inset", type=float, default=0.04,
                    help="每格向内收缩比例，避免切到格线/相邻格")
    ap.add_argument("--width", type=int, default=352,
                    help="统一输出宽度（嘴部锚点 88px 的 4 倍，留足细节）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = Image.open(args.grid).convert("RGBA")
    W, H = src.size
    print(f"输入 {W}x{H}")
    src = kill_bg(src, args.tol)
    cells = split_by_blobs(src, args.rows, args.cols)

    # 关键：9 张必须共用一个坐标系。各自 trim 后分别缩放到同宽，会让「闭嘴」和「大张口」
    # 变成一样大——开口度信息全部丢失。所以统一画布 = 最大 bbox（留边），每张嘴按其
    # 原始尺寸居中放进去，相对大小与高低位置都保持住。
    maxw = max(im.width for _, im in cells)
    maxh = max(im.height for _, im in cells)
    pad = 1.10
    cw2, ch2 = int(maxw * pad), int(maxh * pad)
    scale = args.width / cw2
    ow, oh = args.width, max(1, round(ch2 * scale))
    print(f"最大口型 {maxw}x{maxh} → 统一画布 {cw2}x{ch2} → 输出 {ow}x{oh}")

    meta = {}
    for key, im in cells:
        canvas = Image.new("RGBA", (cw2, ch2), (0, 0, 0, 0))
        # 水平居中；垂直也居中，因为立绘的嘴部锚点是按中心对齐的
        canvas.alpha_composite(im, ((cw2 - im.width) // 2, (ch2 - im.height) // 2))
        outim = canvas.resize((ow, oh), Image.LANCZOS)
        fn = os.path.join(args.out, f"{key}.png")
        outim.save(fn)
        meta[key] = {"contentW": im.width, "contentH": im.height}
        print(f"  {key} {DESC[key]:<12} 原始 {im.width:3d}x{im.height:3d} -> {fn}")

    # 渲染端要知道「画布里嘴实际占多宽」才能正确定标：所有 sprite 共用一张带大量透明
    # 边距的画布，若按画布宽度去贴合嘴部锚点，嘴会被缩掉约 3 倍（实测等效嘴宽 30px
    # vs 原嘴 88px，导致九个口型看起来几乎一样）。maxContentW 是缩放基准。
    info = {
        "canvas": [ow, oh],
        "maxContentW": maxw,
        "maxContentH": maxh,
        "canvasContentW": cw2,      # 输出画布对应的原始画布宽（含 pad）
        "shapes": meta,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"\n写出 meta.json（maxContentW={maxw}，渲染端据此定标）")
    print(f"完成。渲染时加 --visemes {args.out}")


if __name__ == "__main__":
    main()
