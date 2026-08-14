"""Workflow orchestrator — runs all stages in order with checkpointing."""

from deepreact.config.parser import Config
from deepreact.utils.checkpoint import CheckpointManager
from deepreact.utils.logger import WorkflowLogger

from deepreact.workflow.lammps import LammpsStage
from deepreact.workflow.dataset_builder import DatasetBuilderStage
from deepreact.workflow.gaussian import GaussianStage
from deepreact.workflow.check import CheckStage
from deepreact.workflow.dpdata_converter import DpdataStage
from deepreact.workflow.split import SplitStage
from deepreact.workflow.deepmd import DeepmdStage
from deepreact.workflow.active_learning import ActiveLearningLoop


STAGE_ORDER: list[tuple[str, type]] = [
    ("LAMMPS", LammpsStage),
    ("MDDatasetBuilder", DatasetBuilderStage),
    ("Gaussian", GaussianStage),
    ("Convergence Check", CheckStage),
    ("dpdata Conversion", DpdataStage),
    ("Dataset Split", SplitStage),
    ("DeepMD Training", DeepmdStage),
]


class WorkflowManager:
    """Orchestrates the full DeepReact workflow.

    Runs each stage in order. Completed stages are skipped via
    the checkpoint system so that interrupted workflows can resume.
    After the initial model is trained, an active-learning loop
    can optionally refine it.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = WorkflowLogger(
            workdir=config.project.workdir,
            project_name=config.project.name,
        )
        self.checkpoint = CheckpointManager(config.project.workdir)

    def run(self) -> None:
        """Execute all workflow stages in order."""
        project = self.config.project

        self.logger.section(f"DeepReact — {project.name}")
        self.logger.info(f"Work directory: {project.workdir}")
        self.logger.info(f"Log file: {self.logger.file_path}")

        for display_name, stage_cls in STAGE_ORDER:
            try:
                stage = stage_cls(self.config, self.logger, self.checkpoint)
                stage.run()
            except Exception as e:
                self.logger.error(
                    f"\n{'=' * 60}\n"
                    f"  WORKFLOW STOPPED\n"
                    f"  Stage: {display_name}\n"
                    f"  Reason: {e}\n"
                    f"  Check: {self.logger.file_path}\n"
                    f"{'=' * 60}"
                )
                raise SystemExit(1) from e

        # --- active-learning loop (optional) ---
        if self.config.active_learning.enabled:
            try:
                al = ActiveLearningLoop(self.config, self.logger)
                al.run()
            except Exception as e:
                self.logger.error(
                    f"\n{'=' * 60}\n"
                    f"  ACTIVE LEARNING STOPPED\n"
                    f"  Reason: {e}\n"
                    f"  Check: {self.logger.file_path}\n"
                    f"{'=' * 60}"
                )
                raise SystemExit(1) from e

        self.logger.section("WORKFLOW COMPLETE")
        self.logger.info(
            f"All stages finished successfully.\n"
            f"Model: graph.pb (in DeepMD training directory)"
        )
