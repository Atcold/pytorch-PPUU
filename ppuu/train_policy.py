"""Train a policy / controller"""
import os
import json
from collections import defaultdict

import pytorch_lightning as pl
from torch.multiprocessing import set_start_method

from ppuu import lightning_modules
from ppuu import slurm
from ppuu import eval_policy


class CustomLogger(pl.loggers.TensorBoardLogger):
    def __init__(self, *args, json_filename="logs.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
        self.custom_logs = defaultdict(lambda: [])
        self.json_filename = json_filename

    @pl.loggers.base.rank_zero_only
    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)
        self.logs.append(metrics)

    def log_custom(self, key, value):
        self.custom_logs[key].append(value)

    def save(self):
        super().save()
        os.makedirs(self.log_dir, exist_ok=True)
        logs_json_save_path = os.path.join(self.log_dir, self.json_filename)
        dict_to_save = dict(custom=self.custom_logs, logs=self.logs)
        with open(logs_json_save_path, "w") as f:
            json.dump(dict_to_save, f, indent=4)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, run_eval=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_eval = run_eval
        if self.run_eval:
            self.executor = slurm.get_executor(
                "eval_policy", cpus_per_task=8, cluster="slurm"
            )
            # 1 hour is enough for eval
            self.executor.update_parameters(slurm_time="1:00:00")

    def _save_model(self, filepath):
        super()._save_model(filepath)
        if self.run_eval:
            print("evaluating", filepath)
            self.eval_config = eval_policy.EvalConfig(
                checkpoint_path=filepath, save_gradients=True
            )
            self.executor.submit(eval_policy.main, self.eval_config)


def main(config):
    set_start_method("spawn")

    module = lightning_modules.get_module(config.model_config.model_type)

    pl.seed_everything(config.training_config.seed)

    logger = CustomLogger(
        save_dir=config.training_config.output_dir,
        name=config.training_config.experiment_name,
        version=f"seed={config.training_config.seed}",
    )

    period = max(1, config.training_config.n_epochs // 5)
    trainer = pl.Trainer(
        gpus=1,
        gradient_clip_val=50.0,
        max_epochs=config.training_config.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        checkpoint_callback=ModelCheckpoint(
            filepath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=-1,
            save_last=True,
            run_eval=config.training_config.run_eval,
        ),
        logger=logger,
    )
    if config.model_config.checkpoint:
        model = module.load_from_checkpoint(config.model_config.checkpoint)
        model.set_hparams(config)
    else:
        model = module(config)
    trainer.fit(model)
    return model


if __name__ == "__main__":
    print('hasdfa aaaa'   )
    module = lightning_modules.get_module_from_command_line()
    config = module.Config.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(config.training_config.experiment_name)
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)
