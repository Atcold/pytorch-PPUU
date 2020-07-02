import os

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from dataclasses import dataclass
import time

from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import os
from typing import Optional, Callable

import pandas as pd
import gym
import torch
import torch.nn
import torch.nn.parallel
from torch.multiprocessing import Pool, set_start_method

from ppuu import configs
from ppuu import dataloader
from ppuu.lightning_modules.mpur import MPURModule
from ppuu.eval import PolicyEvaluator
from ppuu import slurm

def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


@dataclass
class EvalConfig(configs.ConfigBase):
    checkpoint_path: str = None
    dataset: str = "full"
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: str = None
    test_size_cap: int = None
    slurm: bool = False

    def __post_init__(self):
        if self.num_processes == -1:
            self.num_processes = get_optimal_pool_size()
            logging.info(
                f"Number of processes wasn't speicifed, going to use {self.num_processes}"
            )

        if self.output_dir is None:
            self.checkpoint_path = os.path.normpath(self.checkpoint_path)
            components = self.checkpoint_path.split(os.path.sep)
            components[-2] = "evaluation_results"
            self.output_dir = os.path.join(*components)
            if self.checkpoint_path[0] == os.path.sep:
                self.output_dir = os.path.sep + self.output_dir
            logging.info(
                f"Output dir wasn't specified, going to save to {self.output_dir}"
            )
        if self.dataset in configs.DATASET_PATHS_MAPPING:
            self.dataset = configs.DATASET_PATHS_MAPPING[self.dataset]


def main(config):
    set_start_method("spawn")
    mpur_module = MPURModule.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path
    )

    test_dataset = dataloader.EvaluationDataset(
        config.dataset, "test", config.test_size_cap
    )

    evaluator = PolicyEvaluator(
        test_dataset,
        4,
        build_gradients=config.save_gradients,
        enable_logging=True,
    )
    result = evaluator.evaluate(mpur_module, output_dir=config.output_dir)
    print(result["stats"])


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor('eval', 8)
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)
