import os
import re
import argparse
import subprocess
import h5py
import numpy as np
import cv2


EP_RE = re.compile(r"(episode\d+)\.hdf5$", re.IGNORECASE)


def iter_h5_files(src_dir: str, recursive: bool):
    if recursive:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                if fn.lower().endswith(".hdf5"):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(src_dir):
            if fn.lower().endswith(".hdf5"):
                yield os.path.join(src_dir, fn)


def make_out_path(src_h5: str, src_dir: str, dst_dir: str, keep_subdir: bool, suffix: str = "") -> str:
    base = os.path.basename(src_h5)
    m = EP_RE.search(base)
    stem = m.group(1) if m else os.path.splitext(base)[0]
    out_name = f"{stem}{suffix}.mp4"

    if keep_subdir:
        rel = os.path.relpath(os.path.dirname(src_h5), src_dir)
        out_dir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
    else:
        out_dir = dst_dir

    return os.path.join(out_dir, out_name)


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def transcode_to_h264(src_mp4: str, dst_mp4: str, crf: int = 18, preset: str = "veryfast"):
    cmd = [
        "ffmpeg", "-y",
        "-i", src_mp4,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        "-movflags", "+faststart",
        dst_mp4,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------- decode helpers ----------

def decode_encoded_image(buf) -> np.ndarray:
    """
    For datasets like observation/*/rgb or third_view_rgb: fixed-length encoded bytes (jpg/png).
    Return BGR uint8 image (H,W,3) or None.
    """
    if isinstance(buf, np.bytes_):
        b = bytes(buf)
    elif isinstance(buf, (bytes, bytearray)):
        b = buf
    else:
        try:
            b = buf.tobytes()
        except Exception:
            b = bytes(buf)

    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return img


def normalize_seg_to_bgr(seg: np.ndarray) -> np.ndarray:
    """
    For actor_segmentation/mesh_segmentation: stored as uint8 HxWx3 already.
    Ensure it's BGR uint8 (OpenCV expects BGR). If it's RGB, you can use --swap_rb to swap later.
    """
    if seg is None:
        return None
    if not isinstance(seg, np.ndarray):
        seg = np.array(seg)
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)
    if seg.ndim == 2:
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    if seg.ndim == 3 and seg.shape[-1] == 3:
        return seg
    raise ValueError(f"Unexpected seg shape: {seg.shape}")


# ---------- compose helpers ----------

def compose_3view_frame(top_bgr: np.ndarray, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    Output: 640x720 BGR
      top: 640x480
      left/right: 320x240 bottom
    """
    top = cv2.resize(top_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    left = cv2.resize(left_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_bgr, (320, 240), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((720, 640, 3), dtype=np.uint8)
    canvas[0:480, 0:640] = top
    canvas[480:720, 0:320] = left
    canvas[480:720, 320:640] = right
    return canvas


def maybe_swap_rb(img_bgr: np.ndarray, swap_rb: bool) -> np.ndarray:
    if not swap_rb:
        return img_bgr
    # swap channels (BGR <-> RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def open_writer(path: str, fps: int, size_wh: tuple[int, int], codec: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = size_wh
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {path}. Try --codec mp4v/avc1 or use --to_h264.")
    return writer


# ---------- inspect ----------

def inspect_h5_keys(f: h5py.File, keys: list[str], sample_idx: int = 0):
    print("---- HDF5 Inspect ----")
    for k in keys:
        if k in f:
            ds = f[k]
            print(f"[OK] {k}  shape={ds.shape} dtype={ds.dtype}")
        else:
            print(f"[NO] {k}")
    # try decode one frame for encoded rgb keys
    for k in keys:
        if k in f and k.endswith("/rgb"):
            ds = f[k]
            if len(ds) > sample_idx:
                img = decode_encoded_image(ds[sample_idx])
                if img is None:
                    print(f"    decode {k}[{sample_idx}] -> FAIL")
                else:
                    h, w = img.shape[:2]
                    print(f"    decode {k}[{sample_idx}] -> OK  decoded_size={w}x{h}")
    if "third_view_rgb" in f and len(f["third_view_rgb"]) > sample_idx:
        img = decode_encoded_image(f["third_view_rgb"][sample_idx])
        if img is None:
            print("    decode third_view_rgb -> FAIL")
        else:
            h, w = img.shape[:2]
            print(f"    decode third_view_rgb[{sample_idx}] -> OK  decoded_size={w}x{h}")
    print("----------------------\n")


# ---------- exporters ----------

def export_composed_triplet(
    f: h5py.File,
    out_mp4: str,
    top_key: str,
    left_key: str,
    right_key: str,
    fps: int,
    swap_rb: bool,
    codec: str,
    encoded: bool,
):
    """
    encoded=True: keys are encoded bytes (rgb jpg/png) -> decode via decode_encoded_image
    encoded=False: keys are raw uint8 images (segmentation) -> read array directly
    """
    for k in (top_key, left_key, right_key):
        if k not in f:
            raise KeyError(f"Missing key '{k}'")

    top_ds, left_ds, right_ds = f[top_key], f[left_key], f[right_key]
    T = min(len(top_ds), len(left_ds), len(right_ds))
    if T <= 0:
        raise RuntimeError("No frames in one of datasets.")

    writer = open_writer(out_mp4, fps=fps, size_wh=(640, 720), codec=codec)
    try:
        for t in range(T):
            if encoded:
                top = decode_encoded_image(top_ds[t])
                left = decode_encoded_image(left_ds[t])
                right = decode_encoded_image(right_ds[t])
            else:
                top = normalize_seg_to_bgr(top_ds[t])
                left = normalize_seg_to_bgr(left_ds[t])
                right = normalize_seg_to_bgr(right_ds[t])

            if top is None or left is None or right is None:
                raise RuntimeError(f"Decode/read failed at frame {t}")

            frame = compose_3view_frame(top, left, right)
            frame = maybe_swap_rb(frame, swap_rb)
            writer.write(frame)
    finally:
        writer.release()


def export_single_stream_encoded(
    f: h5py.File,
    out_mp4: str,
    key: str,
    fps: int,
    swap_rb: bool,
    codec: str,
    resize_wh: tuple[int, int] | None,
):
    if key not in f:
        raise KeyError(f"Missing key '{key}'")

    ds = f[key]
    T = len(ds)
    if T <= 0:
        raise RuntimeError("No frames.")

    # Determine output size by decoding one frame
    img0 = decode_encoded_image(ds[0])
    if img0 is None:
        raise RuntimeError("Decode failed at frame 0 for single stream.")

    if resize_wh is None:
        h0, w0 = img0.shape[:2]
        out_w, out_h = w0, h0
    else:
        out_w, out_h = resize_wh

    writer = open_writer(out_mp4, fps=fps, size_wh=(out_w, out_h), codec=codec)
    try:
        for t in range(T):
            img = decode_encoded_image(ds[t])
            if img is None:
                raise RuntimeError(f"Decode failed at frame {t}")

            if resize_wh is not None:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            img = maybe_swap_rb(img, swap_rb)
            writer.write(img)
    finally:
        writer.release()


def main():
    ap = argparse.ArgumentParser("Export composed videos + optional seg/third_view from RoboTwin HDF5.")
    ap.add_argument("--src_dir", type=str, required=True)
    ap.add_argument("--dst_dir", type=str, required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--keep_subdir", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--dry_run", action="store_true")

    # Base RGB keys
    ap.add_argument("--head_rgb", type=str, default="observation/head_camera/rgb")
    ap.add_argument("--left_rgb", type=str, default="observation/left_camera/rgb")
    ap.add_argument("--right_rgb", type=str, default="observation/right_camera/rgb")

    # Optional seg keys
    ap.add_argument("--head_actor", type=str, default="observation/head_camera/actor_segmentation")
    ap.add_argument("--left_actor", type=str, default="observation/left_camera/actor_segmentation")
    ap.add_argument("--right_actor", type=str, default="observation/right_camera/actor_segmentation")

    ap.add_argument("--head_mesh", type=str, default="observation/head_camera/mesh_segmentation")
    ap.add_argument("--left_mesh", type=str, default="observation/left_camera/mesh_segmentation")
    ap.add_argument("--right_mesh", type=str, default="observation/right_camera/mesh_segmentation")

    # Optional arm mask keys
    ap.add_argument("--head_left_arm", type=str, default="observation/head_camera/left_arm_mask")
    ap.add_argument("--left_left_arm", type=str, default="observation/left_camera/left_arm_mask")
    ap.add_argument("--right_left_arm", type=str, default="observation/right_camera/left_arm_mask")

    ap.add_argument("--head_right_arm", type=str, default="observation/head_camera/right_arm_mask")
    ap.add_argument("--left_right_arm", type=str, default="observation/left_camera/right_arm_mask")
    ap.add_argument("--right_right_arm", type=str, default="observation/right_camera/right_arm_mask")

    # Optional third view
    ap.add_argument("--third_key", type=str, default="third_view_rgb")
    ap.add_argument("--third_resize_w", type=int, default=0, help="0 means keep original width")
    ap.add_argument("--third_resize_h", type=int, default=0, help="0 means keep original height")

    # Output options
    ap.add_argument("--include_actor_seg", action="store_true")
    ap.add_argument("--include_mesh_seg", action="store_true")
    ap.add_argument("--include_arm_mask", action="store_true")
    ap.add_argument("--include_third_view", action="store_true")

    # Fix/compat
    ap.add_argument("--swap_rb", action="store_true")
    ap.add_argument("--codec", type=str, default="mp4v")
    ap.add_argument("--inspect", action="store_true", help="Inspect each hdf5 before exporting (prints keys/sizes).")

    # Optional H264 transcode for VSCode/browser
    ap.add_argument("--to_h264", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", type=str, default="veryfast")

    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    if src_dir == dst_dir:
        raise ValueError("src_dir and dst_dir must be different")

    if args.to_h264 and not has_ffmpeg():
        print("[WARN] --to_h264 specified but ffmpeg not found.")
        print("       conda install -c conda-forge ffmpeg")
        return

    h5_files = sorted(list(iter_h5_files(src_dir, args.recursive)))
    if not h5_files:
        print(f"[WARN] No .hdf5 files found under: {src_dir}")
        return

    print(f"[INFO] Found {len(h5_files)} hdf5 files under {src_dir}")

    ok = skip = fail = 0

    for h5_path in h5_files:
        base_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="")
        actor_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_actorseg")
        mesh_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_meshseg")
        left_arm_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_left_arm")
        right_arm_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_right_arm")
        third_out = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_third")

        # Determine which outputs are requested
        planned = [("rgb", base_out)]
        if args.include_actor_seg:
            planned.append(("actor", actor_out))
        if args.include_mesh_seg:
            planned.append(("mesh", mesh_out))
        if args.include_arm_mask:
            planned.append(("left_arm", left_arm_out))
            planned.append(("right_arm", right_arm_out))
        if args.include_third_view:
            planned.append(("third", third_out))

        # skip logic: if all planned exist and no overwrite, skip
        if (not args.overwrite) and all(os.path.exists(p) for _, p in planned):
            print(f"[SKIP] all outputs exist for {os.path.basename(h5_path)}")
            skip += 1
            continue

        print(f"[EP] {h5_path}")

        if args.dry_run:
            for tag, p in planned:
                print(f"  [PLAN] {tag} -> {p}")
            ok += 1
            if args.limit > 0 and (ok + skip + fail) >= args.limit:
                break
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                if args.inspect:
                    keys_to_check = [
                        args.head_rgb, args.left_rgb, args.right_rgb,
                        args.head_actor, args.left_actor, args.right_actor,
                        args.head_mesh, args.left_mesh, args.right_mesh,
                        args.head_left_arm, args.left_left_arm, args.right_left_arm,
                        args.head_right_arm, args.left_right_arm, args.right_right_arm,
                        args.third_key,
                    ]
                    inspect_h5_keys(f, keys_to_check, sample_idx=0)

                # --- RGB composed (always) ---
                if args.to_h264:
                    tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_rgb")
                    export_composed_triplet(
                        f, tmp, args.head_rgb, args.left_rgb, args.right_rgb,
                        fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=True
                    )
                    transcode_to_h264(tmp, base_out, crf=args.crf, preset=args.preset)
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    print(f"  [OK] rgb -> {base_out} (h264)")
                else:
                    export_composed_triplet(
                        f, base_out, args.head_rgb, args.left_rgb, args.right_rgb,
                        fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=True
                    )
                    print(f"  [OK] rgb -> {base_out}")

                # --- actor segmentation composed ---
                if args.include_actor_seg:
                    if args.to_h264:
                        tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_actor")
                        export_composed_triplet(
                            f, tmp, args.head_actor, args.left_actor, args.right_actor,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        transcode_to_h264(tmp, actor_out, crf=args.crf, preset=args.preset)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        print(f"  [OK] actor_seg -> {actor_out} (h264)")
                    else:
                        export_composed_triplet(
                            f, actor_out, args.head_actor, args.left_actor, args.right_actor,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        print(f"  [OK] actor_seg -> {actor_out}")

                # --- mesh segmentation composed ---
                if args.include_mesh_seg:
                    if args.to_h264:
                        tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_mesh")
                        export_composed_triplet(
                            f, tmp, args.head_mesh, args.left_mesh, args.right_mesh,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        transcode_to_h264(tmp, mesh_out, crf=args.crf, preset=args.preset)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        print(f"  [OK] mesh_seg -> {mesh_out} (h264)")
                    else:
                        export_composed_triplet(
                            f, mesh_out, args.head_mesh, args.left_mesh, args.right_mesh,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        print(f"  [OK] mesh_seg -> {mesh_out}")

                # --- arm mask composed ---
                if args.include_arm_mask:
                    # Left Arm
                    if args.to_h264:
                        tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_left_arm")
                        export_composed_triplet(
                            f, tmp, args.head_left_arm, args.left_left_arm, args.right_left_arm,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        transcode_to_h264(tmp, left_arm_out, crf=args.crf, preset=args.preset)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        print(f"  [OK] left_arm -> {left_arm_out} (h264)")
                    else:
                        export_composed_triplet(
                            f, left_arm_out, args.head_left_arm, args.left_left_arm, args.right_left_arm,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        print(f"  [OK] left_arm -> {left_arm_out}")

                    # Right Arm
                    if args.to_h264:
                        tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_right_arm")
                        export_composed_triplet(
                            f, tmp, args.head_right_arm, args.left_right_arm, args.right_right_arm,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        transcode_to_h264(tmp, right_arm_out, crf=args.crf, preset=args.preset)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        print(f"  [OK] right_arm -> {right_arm_out} (h264)")
                    else:
                        export_composed_triplet(
                            f, right_arm_out, args.head_right_arm, args.left_right_arm, args.right_right_arm,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, encoded=False
                        )
                        print(f"  [OK] right_arm -> {right_arm_out}")

                # --- third view (single stream) ---
                if args.include_third_view:
                    resize_wh = None
                    if args.third_resize_w > 0 and args.third_resize_h > 0:
                        resize_wh = (args.third_resize_w, args.third_resize_h)

                    if args.to_h264:
                        tmp = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir, suffix="_tmp_third")
                        export_single_stream_encoded(
                            f, tmp, args.third_key,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, resize_wh=resize_wh
                        )
                        transcode_to_h264(tmp, third_out, crf=args.crf, preset=args.preset)
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                        print(f"  [OK] third_view -> {third_out} (h264)")
                    else:
                        export_single_stream_encoded(
                            f, third_out, args.third_key,
                            fps=args.fps, swap_rb=args.swap_rb, codec=args.codec, resize_wh=resize_wh
                        )
                        print(f"  [OK] third_view -> {third_out}")

            ok += 1

        except Exception as e:
            print(f"  [FAIL] {e}")
            fail += 1

        if args.limit > 0 and (ok + skip + fail) >= args.limit:
            break

    print(f"\n[SUMMARY] ok={ok} skip={skip} fail={fail}  dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
