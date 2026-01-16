import pickle
import torch
import numpy as np
from typing import Any


PKL_PATH = "/data/dex/RoboTwin/data/blocks_ranking_rgb/demo_clean/_traj_data/episode0.pkl"


def is_tensor_like(x):
    return isinstance(x, (torch.Tensor, np.ndarray))


def tensor_info(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor shape={tuple(x.shape)}, dtype={x.dtype}"
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape}, dtype={x.dtype}"
    return ""


def inspect(obj: Any, prefix: str = "", max_depth: int = 6, depth: int = 0, max_items: int = 10):
    """
    Recursively inspect object structure.
    """
    if depth > max_depth:
        print(f"{prefix}: <Max depth reached>")
        return

    # Tensor / ndarray
    if is_tensor_like(obj):
        print(f"{prefix}: {tensor_info(obj)}")
        return

    # Dict
    if isinstance(obj, dict):
        print(f"{prefix}: dict (keys={list(obj.keys())})")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}.{k}: <Skipped remaining items>")
                break
            inspect(v, f"{prefix}.{k}" if prefix else str(k), max_depth, depth + 1)
        return

    # List / Tuple
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__} (len={len(obj)})")
        for i, v in enumerate(obj[:max_items]):
            inspect(v, f"{prefix}[{i}]", max_depth, depth + 1)
        if len(obj) > max_items:
            print(f"{prefix}: <Skipped remaining items>")
        return

    # Custom object
    if hasattr(obj, "__dict__"):
        attrs = list(obj.__dict__.keys())
        print(f"{prefix}: object {type(obj).__name__} (attrs={attrs})")
        for k in attrs[:max_items]:
            inspect(getattr(obj, k), f"{prefix}.{k}", max_depth, depth + 1)
        return

    # Fallback
    print(f"{prefix}: {type(obj).__name__} = {repr(obj)}")


def main():
    print(f"Loading: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    print("\n===== TOP LEVEL =====")
    print(f"Type: {type(data)}")

    print("\n===== STRUCTURE =====")
    inspect(data)

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
