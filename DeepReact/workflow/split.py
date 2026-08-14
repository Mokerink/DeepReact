"""Dataset splitting stage.

If ``scripts.split`` is set the external script is run; otherwise a
built-in splitter shuffles all ``set.xxx`` directories and partitions
them into train / validation / test subsets.  Ratios and seed are read
from the optional ``split`` config section.
"""

import random
import shutil

from deepreact.workflow.base import BaseStage


class SplitStage(BaseStage):
    """Randomly split DeepMD datasets into train / val / test."""

    stage_name = "split"

    def execute(self) -> None:
        if self.config.scripts.split:
            self._run_external()
        else:
            self._run_builtin()

    # -- external script ------------------------------------------------
    def _run_external(self) -> None:
        import shlex
        cmd = shlex.split(self.config.scripts.split)
        self._run_cmd(cmd, cwd=self.config.config_dir)
        self.logger.info("Dataset splitting complete.")

    # -- built-in -------------------------------------------------------
    def _run_builtin(self) -> None:
        split_cfg = self.config.split
        src_dir = self.config.config_dir / "deepmd_data"

        all_sets = sorted(src_dir.glob("set.[0-9][0-9][0-9]"))
        all_sets = [d for d in all_sets if d.is_dir()]
        total = len(all_sets)

        if total == 0:
            self.logger.info(f"No set.* directories found in {src_dir}")
            return

        random.seed(split_cfg.seed)
        shuffled = all_sets.copy()
        random.shuffle(shuffled)

        n_train = int(total * split_cfg.train_ratio)
        n_val = int(total * split_cfg.val_ratio)

        train_sets = shuffled[:n_train]
        val_sets = shuffled[n_train:n_train + n_val]
        test_sets = shuffled[n_train + n_val:]

        self.logger.info(
            f"Split {total} sets -> "
            f"train: {len(train_sets)} ({len(train_sets)/total*100:.0f}%), "
            f"val:   {len(val_sets)} ({len(val_sets)/total*100:.0f}%), "
            f"test:  {len(test_sets)} ({len(test_sets)/total*100:.0f}%)"
        )

        targets = {
            "train": train_sets,
            "val": val_sets,
            "test": test_sets,
        }

        for name, folders in targets.items():
            dest_dir = src_dir / name
            dest_dir.mkdir(exist_ok=True)
            for src in folders:
                dst = dest_dir / src.name
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(src), str(dst))
            self.logger.info(f"  {name}: {len(folders)} sets -> {dest_dir}")

        self.logger.info("Dataset splitting complete.")
