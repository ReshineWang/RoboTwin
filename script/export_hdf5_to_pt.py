import os
import re
import argparse
import h5py
import numpy as np
import torch


EP_RE = re.compile(r"(episode\d+)\.hdf5$", re.IGNORECASE)


def load_action_vector(h5_path: str, key: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {h5_path}. Top keys: {list(f.keys())}")
        arr = f[key][...]
    # ensure float32 tensor
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return torch.from_numpy(arr).to(torch.float32).contiguous()


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


def make_out_path(src_h5: str, src_dir: str, dst_dir: str, keep_subdir: bool) -> str:
    """
    keep_subdir=True: preserve relative path under src_dir (e.g., a/b/episode0.hdf5 -> dst/a/b/episode0.pt)
    keep_subdir=False: flatten to dst/episode0.pt (collisions possible if duplicate names)
    """
    base = os.path.basename(src_h5)
    m = EP_RE.search(base)
    out_name = (m.group(1) if m else os.path.splitext(base)[0]) + ".pt"

    if keep_subdir:
        rel = os.path.relpath(os.path.dirname(src_h5), src_dir)
        out_dir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
    else:
        out_dir = dst_dir

    return os.path.join(out_dir, out_name)


def main():
    parser = argparse.ArgumentParser("Batch export RoboTwin action vectors from episode*.hdf5 to .pt")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing episode*.hdf5")
    parser.add_argument("--dst_dir", type=str, required=True, help="Directory to save episode*.pt")
    parser.add_argument("--action_key", type=str, default="joint_action/vector",
                        help="HDF5 dataset path for action vector (default: joint_action/vector)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .hdf5 files")
    parser.add_argument("--keep_subdir", action="store_true",
                        help="Preserve subfolder structure under src_dir in dst_dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    parser.add_argument("--dry_run", action="store_true", help="Only print conversions, do not write files")
    parser.add_argument("--limit", type=int, default=-1, help="Convert at most N files (debug). -1 means no limit")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    if src_dir == dst_dir:
        raise ValueError("src_dir and dst_dir must be different")

    h5_files = sorted(list(iter_h5_files(src_dir, args.recursive)))
    if not h5_files:
        print(f"[WARN] No .hdf5 files found under: {src_dir}")
        return

    print(f"[INFO] Found {len(h5_files)} hdf5 files under {src_dir}")
    ok, skip, fail = 0, 0, 0

    for i, h5_path in enumerate(h5_files):
        if args.limit > 0 and ok + skip + fail >= args.limit:
            break

        out_pt = make_out_path(h5_path, src_dir, dst_dir, keep_subdir=args.keep_subdir)
        if os.path.exists(out_pt) and not args.overwrite:
            print(f"[SKIP] exists: {out_pt}")
            skip += 1
            continue

        print(f"[CONVERT] {h5_path} -> {out_pt}")

        if args.dry_run:
            ok += 1
            continue

        try:
            action = load_action_vector(h5_path, key=args.action_key)
            os.makedirs(os.path.dirname(out_pt), exist_ok=True)
            torch.save(action, out_pt)
            print(f"  [OK] shape={tuple(action.shape)} dtype={action.dtype}")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            fail += 1

    print(f"\n[SUMMARY] ok={ok} skip={skip} fail={fail}  dst_dir={dst_dir}")


if __name__ == "__main__":
    main()
