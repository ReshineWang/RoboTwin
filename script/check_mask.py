import argparse
import os
import math
import h5py
import numpy as np
import cv2


def decode_rgb_bytes(buf) -> np.ndarray:
    """Decode encoded image bytes stored in HDF5 (jpg/png). Return BGR uint8."""
    if isinstance(buf, np.bytes_):
        buf = bytes(buf)
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return img


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def mask_from_color(seg_hwc: np.ndarray, color_rgb: np.ndarray) -> np.ndarray:
    """seg_hwc: uint8 HxWx3, color_rgb: uint8 [3]"""
    return (seg_hwc == color_rgb).all(axis=-1)


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Overlay mask on BGR image.
    Highlight masked pixels in red (BGR: 0,0,255).
    """
    out = bgr.copy()
    red = np.zeros_like(out)
    red[..., 2] = 255
    m = mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + red[m].astype(np.float32) * alpha).astype(np.uint8)
    return out


def put_text(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_grid(images, cols=5, pad=8, bg=30):
    """images: list of same-size BGR images. Return grid BGR."""
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    grid_h = rows * h + (rows + 1) * pad
    grid_w = cols * w + (cols + 1) * pad
    canvas = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)

    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = im
    return canvas


def parse_color_list(tokens):
    """
    tokens example: ["0,100,0", "75,0,130"]
    return list of np.uint8([r,g,b])
    """
    colors = []
    for t in tokens:
        parts = [int(x) for x in t.split(",")]
        assert len(parts) == 3
        colors.append(np.array(parts, dtype=np.uint8))
    return colors


def main():
    ap = argparse.ArgumentParser("Visualize actor_segmentation colors as masks/overlays.")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cam", type=str, default="head_camera")
    ap.add_argument("--topk", type=int, default=40, help="Top-k colors by pixel count to visualize")
    ap.add_argument("--min_ratio", type=float, default=0.0005, help="Ignore colors with pixel ratio < min_ratio")
    ap.add_argument("--save_dir", type=str, default="seg_vis")
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha")
    ap.add_argument("--grid_cols", type=int, default=5)

    # optional: only visualize specified colors
    ap.add_argument("--pick_colors", nargs="*", default=[],
                    help='Space-separated colors like: 0,100,0 75,0,130 (if set, ignores topk)')

    args = ap.parse_args()

    seg_key = f"observation/{args.cam}/actor_segmentation"
    rgb_key = f"observation/{args.cam}/rgb"

    ensure_dir(args.save_dir)

    with h5py.File(args.h5, "r") as f:
        if seg_key not in f:
            raise KeyError(f"Missing key: {seg_key}")
        if rgb_key not in f:
            raise KeyError(f"Missing key: {rgb_key}")

        seg = f[seg_key][args.frame]  # uint8 HWC (RGB-like labeling)
        rgb_bgr = decode_rgb_bytes(f[rgb_key][args.frame])

    # Save raw rgb for reference
    rgb_path = os.path.join(args.save_dir, f"rgb_{args.cam}_f{args.frame}.png")
    cv2.imwrite(rgb_path, rgb_bgr)
    print(f"[OK] saved rgb: {rgb_path}")

    H, W = seg.shape[:2]
    flat = seg.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)

    # choose colors
    chosen = []
    if args.pick_colors:
        chosen = parse_color_list(args.pick_colors)
    else:
        for idx in order[:args.topk]:
            c = colors[idx]
            ratio = counts[idx] / (H * W)
            if ratio < args.min_ratio:
                continue
            chosen.append(c)

    print(f"[INFO] seg={seg_key} frame={args.frame} unique_colors={len(colors)} chosen={len(chosen)}")

    overlay_images_for_grid = []

    for rank, c in enumerate(chosen):
        c_list = c.tolist()
        mask = mask_from_color(seg, c)
        ratio = float(mask.mean())

        # save mask
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_path = os.path.join(args.save_dir, f"mask_{rank:02d}_{c_list[0]}-{c_list[1]}-{c_list[2]}.png")
        cv2.imwrite(mask_path, mask_u8)

        # overlay
        ov = overlay_mask(rgb_bgr, mask, alpha=args.alpha)
        label = f"{rank:02d} color={c_list} ratio={ratio:.4f}"
        ov = put_text(ov, label)
        ov_path = os.path.join(args.save_dir, f"overlay_{rank:02d}_{c_list[0]}-{c_list[1]}-{c_list[2]}.png")
        cv2.imwrite(ov_path, ov)

        overlay_images_for_grid.append(ov)

        print(f"[OK] {rank:02d} color={c_list} ratio={ratio:.4f} -> {mask_path} , {ov_path}")

    # grid
    grid = make_grid(overlay_images_for_grid, cols=args.grid_cols)
    if grid is not None:
        grid_path = os.path.join(args.save_dir, f"grid_top{len(overlay_images_for_grid)}_{args.cam}_f{args.frame}.png")
        cv2.imwrite(grid_path, grid)
        print(f"[OK] saved grid: {grid_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
