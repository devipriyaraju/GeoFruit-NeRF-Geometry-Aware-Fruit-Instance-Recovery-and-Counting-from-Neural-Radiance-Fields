import argparse
import json
from pathlib import Path

import numpy as np

from fruitnerf_repro.counting import count_fruits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cloud", type=Path, help="NPZ file that contains fruit_pts and spacing")
    p.add_argument("--config", type=Path, default=Path("configs/frozen_counting_params.json"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    data = np.load(args.cloud)
    spacing = float(data["spacing"])
    count, centers, info, _, _ = count_fruits(data["fruit_pts"], spacing, cfg)
    print(json.dumps({"count": count, "info": info}, indent=2))
    out = args.cloud.with_name(args.cloud.stem + "_centers.npy")
    np.save(out, centers)
    print(f"saved centers to {out}")


if __name__ == "__main__":
    main()
