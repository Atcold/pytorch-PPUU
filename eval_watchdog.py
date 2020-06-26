import os
import argparse
import re
import time
import slurm
import glob
import eval_policy

MODEL_REGEX = ".*policy_networks.*step\d+.model$"

already_run = ["dreaming_uptrain", "fixed_eval_4"]


def submit(executor, path):
    print("submitting", path)
    if path.endswith('=0.ckpt'):
        return None
    config = eval_policy.EvalConfig(
        checkpoint_path=path, save_gradients=True, num_processes=10
    )
    if not os.path.exists(config.output_dir):
        return executor.submit(eval_policy.main, config)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dir", type=str, default=".")
    parser.add_argument(
        "--check_interval",
        type=int,
        default=300,
        help="interval in seconds between checks for new results",
    )
    parser.add_argument("--cluster", type=str, default="slurm")
    opt = parser.parse_args()

    executor = slurm.get_executor(
        job_name="eval", cpus_per_task=8, cluster=opt.cluster
    )
    self.executor.update_parameters(slurm_time="1:00:00")

    path_regex = os.path.join(opt.dir, "**/*.ckpt")
    print(path_regex)

    while True:
        checkpoints = glob.glob(path_regex, recursive=True)
        for checkpoint in checkpoints:
            if checkpoint not in already_run:
                already_run.append(checkpoint)
                job = submit(executor, checkpoint)
                if job is not None and opt.cluster in ["local", "debug"]:
                    print(job.result())
        print("done")
        time.sleep(opt.check_interval)


if __name__ == "__main__":
    main()
