"""Asset locations for the lightweight portable API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VALID_PESTS: frozenset[str] = frozenset({
    "BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
    "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight",
})

# Mirrors api_handoff_transformer/run_predict.py:66 — the deployed contract value.
MODEL_VERSION = "v0-transformer-real-ckpt"


@dataclass(frozen=True)
class Paths:
    pkg_root: Path
    input_dir: Path
    output_dir: Path

    @property
    def assets(self) -> Path:
        return self.pkg_root / "assets"

    @property
    def stage1_dir(self) -> Path:
        return self.assets / "stage1"

    @property
    def stage2_dir(self) -> Path:
        return self.assets / "stage2"

    @property
    def configs_dir(self) -> Path:
        return self.assets / "configs"

    @property
    def climatology_dir(self) -> Path:
        return self.assets / "climatology"

    def stage1_pest(self, pest: str) -> Path:
        return self.stage1_dir / pest

    def stage2_pest(self, pest: str) -> Path:
        return self.stage2_dir / pest

    def climatology_csv(self, pest: str) -> Path:
        return self.climatology_dir / f"{pest}_climatology_train_stats.csv"

    @staticmethod
    def from_root(pkg_root: Path, input_dir: Path | None = None,
                  output_dir: Path | None = None) -> "Paths":
        pkg_root = Path(pkg_root).resolve()
        return Paths(
            pkg_root=pkg_root,
            input_dir=Path(input_dir) if input_dir else pkg_root / "input",
            output_dir=Path(output_dir) if output_dir else pkg_root / "output",
        )
