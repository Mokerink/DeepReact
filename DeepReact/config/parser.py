"""YAML configuration parser for DeepReact workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str
    workdir: Path


@dataclass
class LammpsConfig:
    executable: str
    input: Path
    output: Path
    bonds: Path
    cwd: Path | None = None


@dataclass
class MddatasetbuilderConfig:
    trajectory: Path
    bonds: Path | None = None
    output: Path = Path(".")
    atoms: str = "C H O"
    name: str = "mydata"
    cutoff: float = 2.0
    stride: int = 1000
    nprocjob: int = 64
    keywords: str = "force b3lyp/3-21G* Geom=PrintInputOrient"
    gjf_inject: str | None = None
    command: str | None = None


@dataclass
class GaussianConfig:
    mode: str  # "run" or "export"
    executable: str
    input_dir: Path
    output_dir: Path
    template: Path | None = None


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 42

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 4)


@dataclass
class ScriptsConfig:
    check: str = ""
    split: str = ""
    dpdata: str = ""


@dataclass
class DeepmdConfig:
    train: Path
    executable: str = "dp"


@dataclass
class ActiveLearningConfig:
    enabled: bool = False
    lammps_input: Path | None = None
    lammps_data: Path | None = None
    trajectory_output: str = "deepmd.lammpstrj"
    gjf_output: str = "./gaussian_al"
    log_output: str = "./gaussian_al/log"
    dpdata_output: str = "./deepmd_data_al"
    dptest_model: str = "graph.pb"
    dptest_system: str = "./deepmd_data_al"
    dptest_output: str = "./dptest_out"
    energy_rmse_threshold: float = 0.01   # eV — mark if above
    force_rmse_threshold: float = 0.1     # eV/Å — mark if above
    iterations: int = 1
    train_json: Path | None = None
    checkpoint: str = "./model.ckpt"


@dataclass
class Config:
    project: ProjectConfig
    lammps: LammpsConfig
    mddatasetbuilder: MddatasetbuilderConfig
    gaussian: GaussianConfig
    scripts: ScriptsConfig
    deepmd: DeepmdConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    config_dir: Path = field(default=Path("."))
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(config_path: str | Path) -> Config:
    """Load and parse a DeepReact YAML configuration file."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config_dir = config_path.parent

    _validate_required_sections(raw)

    project = ProjectConfig(
        name=raw["project"]["name"],
        workdir=config_dir / raw["project"]["workdir"],
    )

    lammps = LammpsConfig(
        executable=raw["lammps"]["executable"],
        input=_resolve(config_dir, raw["lammps"]["input"]),
        output=_resolve(config_dir, raw["lammps"]["output"]),
        bonds=_resolve(config_dir, raw["lammps"]["bonds"]),
        cwd=_resolve(config_dir, raw["lammps"]["cwd"])
        if raw["lammps"].get("cwd")
        else None,
    )

    mdb = MddatasetbuilderConfig(
        trajectory=_resolve(config_dir, raw["mddatasetbuilder"]["trajectory"]),
        bonds=_resolve(config_dir, raw["mddatasetbuilder"]["bonds"])
        if raw["mddatasetbuilder"].get("bonds")
        else None,
        output=_resolve(config_dir, raw["mddatasetbuilder"]["output"])
        if raw["mddatasetbuilder"].get("output")
        else Path("."),
        atoms=raw["mddatasetbuilder"].get("atoms", "C H O"),
        name=raw["mddatasetbuilder"].get("name", "mydata"),
        cutoff=float(raw["mddatasetbuilder"].get("cutoff", 2.0)),
        stride=int(raw["mddatasetbuilder"].get("stride", 1000)),
        nprocjob=int(raw["mddatasetbuilder"].get("nprocjob", 64)),
        keywords=raw["mddatasetbuilder"].get(
            "keywords", "force b3lyp/3-21G* Geom=PrintInputOrient"
        ),
        gjf_inject=raw["mddatasetbuilder"].get("gjf_inject"),
        command=raw["mddatasetbuilder"].get("command"),
    )

    gaussian = GaussianConfig(
        mode=raw["gaussian"]["mode"],
        executable=raw["gaussian"]["executable"],
        input_dir=_resolve(config_dir, raw["gaussian"]["input_dir"]),
        output_dir=_resolve(config_dir, raw["gaussian"]["output_dir"]),
        template=_resolve(config_dir, raw["gaussian"]["template"])
        if raw["gaussian"].get("template")
        else None,
    )

    scripts = ScriptsConfig(
        check=raw.get("scripts", {}).get("check", ""),
        split=raw.get("scripts", {}).get("split", ""),
        dpdata=raw.get("scripts", {}).get("dpdata", ""),
    )

    split_cfg = raw.get("split", {})
    split = SplitConfig(
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        seed=int(split_cfg.get("seed", 42)),
    )

    deepmd = DeepmdConfig(
        train=_resolve(config_dir, raw["deepmd"]["train"]),
        executable=raw["deepmd"].get("executable", "dp"),
    )

    al_cfg = raw.get("active_learning", {})
    active_learning = ActiveLearningConfig(
        enabled=bool(al_cfg.get("enabled", False)),
        lammps_input=_resolve(config_dir, al_cfg["lammps_input"])
        if al_cfg.get("lammps_input")
        else None,
        lammps_data=_resolve(config_dir, al_cfg["lammps_data"])
        if al_cfg.get("lammps_data")
        else None,
        trajectory_output=al_cfg.get("trajectory_output", "deepmd.lammpstrj"),
        gjf_output=al_cfg.get("gjf_output", "./gaussian_al"),
        log_output=al_cfg.get("log_output", "./gaussian_al/log"),
        dpdata_output=al_cfg.get("dpdata_output", "./deepmd_data_al"),
        dptest_model=al_cfg.get("dptest_model", "graph.pb"),
        dptest_system=al_cfg.get("dptest_system", "./deepmd_data_al"),
        dptest_output=al_cfg.get("dptest_output", "./dptest_out"),
        energy_rmse_threshold=float(al_cfg.get("energy_rmse_threshold", 0.01)),
        force_rmse_threshold=float(al_cfg.get("force_rmse_threshold", 0.1)),
        iterations=int(al_cfg.get("iterations", 1)),
        train_json=_resolve(config_dir, al_cfg["train_json"])
        if al_cfg.get("train_json")
        else None,
        checkpoint=al_cfg.get("checkpoint", "./model.ckpt"),
    )

    return Config(
        project=project,
        lammps=lammps,
        mddatasetbuilder=mdb,
        gaussian=gaussian,
        scripts=scripts,
        deepmd=deepmd,
        split=split,
        active_learning=active_learning,
        config_dir=config_dir,
        _raw=raw,
    )


def _resolve(config_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def _validate_required_sections(raw: dict) -> None:
    required = ["project", "lammps", "mddatasetbuilder", "gaussian", "deepmd"]
    missing = [s for s in required if s not in raw]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    project = raw["project"]
    if "name" not in project:
        raise ValueError("project.name is required")
    if "workdir" not in project:
        raise ValueError("project.workdir is required")
