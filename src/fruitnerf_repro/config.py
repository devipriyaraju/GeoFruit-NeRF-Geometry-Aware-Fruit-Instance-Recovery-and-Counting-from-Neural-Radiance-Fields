from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    tree: str = "tree_01"
    semantic_dir: str = "semantics_sam"
    preset: str = "fruit_nerf"
    downscale: int = 4
    export_res: int = 512
    density_thr: float = 150.0
    semantic_thr: float = 0.9
    outlier_radius_mult: float = 3.0
    outlier_min_nb: int = 10
    dbscan_eps_mult: float = 2.0
    dbscan_min_samples: int = 10
    min_size_factor: float = 0.3
    multi_factor: float = 1.6
    max_fruits_per_cluster: int = 6
    multi_mode: str = "hausdorff"
    seed: int = 42
    data_root: Path = Path("data/external/FruitNeRF_Real")
    out_root: Path = Path("outputs")


GT_COUNTS = {
    "tree_01": 179,
    "tree_02": 113,
    "tree_03": 291,
}

PAPER_SAM_COUNTS = {
    "tree_01": 147,
    "tree_02": 86,
    "tree_03": 190,
}
