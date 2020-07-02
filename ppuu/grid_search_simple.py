import time
import os
import argparse
import numpy as np

import pytorch_lightning as pl

from ppuu import slurm

from ppuu.lightning_modules import MPURKMSplitModule as Module
from ppuu.train_policy import CustomLogger
from ppuu import configs

EPOCHS = 21

from torch.multiprocessing import set_start_method


def generate_config():
    config = Module.Config()
    config.cost_config.lambda_l = 10 ** np.random.uniform(-3, 1)
    config.cost_config.lambda_p = (
        config.cost_config.lambda_l * np.random.uniform(0.1, 4) ** 2
    )
    config.cost_config.lambda_o = (
        config.cost_config.lambda_l * np.random.uniform(0.1, 2) ** 2
    )
    config.cost_config.u_reg = 0.05
    config.cost_config.lambda_a = (
        config.cost_config.lambda_l * np.random.uniform(0, 1) ** 2
    )
    config.cost_config.agg_func_str = f"logsumexp-{np.random.randint(15, 85)}"
    config.cost_config.masks_power = np.random.uniform(1, 10)
    return config


def run_trial(output_dir):
    config = generate_config()
    config.training_config.n_epochs = EPOCHS + 1
    config.training_config.epoch_size = 500
    config.training_config.validation_size = 10
    # config.training_config.dataset = configs.DATASET_PATHS_MAPPING[""]
    config.cost_config.uncertainty_n_batches = 100
    exp_name = f"grid_search_{time.time()}"
    for seed in [1, 2, 3]:
        config.training_config.seed = seed
        logger = CustomLogger(
            save_dir=output_dir,
            name=exp_name,
            version=f"seed={config.training_config.seed}",
        )
        trainer = pl.Trainer(
            gpus=1,
            gradient_clip_val=50.0,
            max_epochs=config.training_config.n_epochs,
            check_val_every_n_epoch=EPOCHS,
            num_sanity_val_steps=0,
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(logger.log_dir, "checkpoints"),
                save_top_k=-1,
                save_last=True,
            ),
            logger=logger,
        )
        model = Module(config)
        trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output dir"
    )
    args = parser.parse_args()
    set_start_method("spawn")

    for i in range(1000):
        run_trial(args.output_dir)
