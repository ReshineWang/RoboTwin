import os
import argparse
import h5py
import numpy as np


def summarize_array(arr: np.ndarray, max_elems: int = 20) -> str:
    """Small summary for numeric arrays."""
    if arr.size == 0:
        return "empty"
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1)
        sample = flat[:max_elems]
        return (f"min={np.min(flat):.4g}, max={np.max(flat):.4g}, "
                f"mean={np.mean(flat):.4g}, std={np.std(flat):.4g}, "
                f"sample={sample}")
    else:
        flat = arr.reshape(-1)
        sample = flat[:max_elems]
        return f"sample={sample}"


def is_probably_image_bitstream(dset: h5py.Dataset) -> bool:
    """
    RoboTwin doc says images can be stored as bitstream in HDF5.
    Common patterns: dtype=uint8 and 1D variable length, or dtype=object/bytes.
    """
    dt = dset.dtype
    if dt == np.uint8 and (dset.ndim == 1 or dset.ndim == 2):
        # Could be raw bytes buffers.
        return True
    if dt.kind in ["S", "O"]:  # fixed-length bytes or object (vlen)
        return True
    # h5py special vlen bytes
    try:
        if h5py.check_dtype(vlen=dt) is bytes:
            return True
    except Exception:
        pass
    return False


def print_attrs(h5obj, indent: str):
    if len(h5obj.attrs) == 0:
        return
    print(f"{indent}attrs:")
    for k in h5obj.attrs.keys():
        v = h5obj.attrs[k]
        # convert bytes to readable
        if isinstance(v, (bytes, np.bytes_)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                pass
        print(f"{indent}  - {k}: {v}")


def walk(name: str, obj, max_preview_bytes: int, max_preview_elems: int):
    indent = "  " * (name.count("/") if name else 0)

    if isinstance(obj, h5py.Group):
        print(f"{indent}[GROUP] {name if name else '/'}")
        print_attrs(obj, indent + "  ")

    elif isinstance(obj, h5py.Dataset):
        comp = obj.compression
        chunks = obj.chunks
        dt = obj.dtype
        shape = obj.shape

        print(f"{indent}[DSET ] {name}  shape={shape}  dtype={dt}  compression={comp}  chunks={chunks}")
        print_attrs(obj, indent + "  ")

        # preview
        try:
            if obj.size == 0:
                return

            # If it's likely image bitstream, only show length hints
            if is_probably_image_bitstream(obj):
                # read a small slice safely
                if obj.ndim == 1:
                    item0 = obj[0]
                    if isinstance(item0, (bytes, np.bytes_)):
                        print(f"{indent}  preview: bytes_len={len(item0)} (likely encoded image/video frame)")
                    elif isinstance(item0, np.ndarray) and item0.dtype == np.uint8:
                        print(f"{indent}  preview: uint8 buffer len={item0.size} (likely encoded image)")
                    else:
                        print(f"{indent}  preview: type(item0)={type(item0)}")
                else:
                    # 2D/3D: show first row sizes
                    arr = obj[0]
                    if isinstance(arr, np.ndarray):
                        print(f"{indent}  preview: first_item shape={arr.shape}, dtype={arr.dtype}")
                print(f"{indent}  hint: decode with cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)")
                return

            # numeric / general small datasets preview
            # Avoid loading huge datasets
            if obj.size <= max_preview_elems:
                arr = obj[...]
                if isinstance(arr, np.ndarray):
                    print(f"{indent}  stats: {summarize_array(arr, max_elems=min(max_preview_elems, 20))}")
            else:
                # read a small head slice
                if obj.ndim == 1:
                    arr = obj[:min(max_preview_elems, obj.shape[0])]
                elif obj.ndim >= 2:
                    head0 = min(max_preview_elems, obj.shape[0])
                    # take first few on axis0 only to limit memory
                    slicing = (slice(0, head0),) + tuple(slice(0, min(5, s)) for s in obj.shape[1:])
                    arr = obj[slicing]
                else:
                    arr = obj[...]
                if isinstance(arr, np.ndarray):
                    print(f"{indent}  head preview: shape={arr.shape}, {summarize_array(arr, max_elems=20)}")

        except Exception as e:
            print(f"{indent}  preview_error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True, help="Path to episodeX.hdf5")
    parser.add_argument("--max_preview_elems", type=int, default=200, help="Max elems to fully load for preview")
    parser.add_argument("--max_preview_bytes", type=int, default=4096, help="(reserved) bytes preview cap")
    args = parser.parse_args()

    h5_path = args.h5
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    print(f"Opening HDF5: {h5_path}\n")
    with h5py.File(h5_path, "r") as f:
        # Print root attrs
        print("[ROOT] /")
        print_attrs(f, "  ")
        print()

        # Walk recursively
        f.visititems(lambda name, obj: walk(name, obj, args.max_preview_bytes, args.max_preview_elems))

    print("\nDone.")


if __name__ == "__main__":
    main()
