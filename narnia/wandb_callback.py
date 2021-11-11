import functools
import importlib.util
import numbers
import os
import sys
import tempfile
import wandb
from pathlib import Path

from transformers.file_utils import is_datasets_available
from transformers.utils import logging
from transformers.file_utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
from transformers.trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from transformers.integrations import WandbCallback


def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


logger = logging.get_logger(__name__)


def rewrite_logs(d, prefix):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d[prefix + "-eval/" + k[eval_prefix_len:]] = v
        else:
            new_d[prefix + "-train/" + k] = v
    return new_d


class WandbPrefixCallback(WandbCallback):
    def __init__(self, prefix):
        self.prefix = prefix
        super().__init__()

    def setup(self, args, state, model, **kwargs):
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric(f"{self.prefix}-train/global_step")
                self._wandb.define_metric("*", step_metric=f"{self.prefix}-train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"{self.prefix}-eval/{args.metric_for_best_model}": state.best_metric,
                        f"{self.prefix}-train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs, self.prefix)
            self._wandb.log({**logs, f"{self.prefix}-train/global_step": state.global_step})


class SBERTWandbCallback:
    def __init__(self, prefix):
        self.prefix = prefix
    
    def log(self, score, epoch, steps):
        wandb.log({
            f"{self.prefix}-eval/step": steps,
            f"{self.prefix}-eval/epoch": epoch,
            f"{self.prefix}-eval/score": score,
        })
