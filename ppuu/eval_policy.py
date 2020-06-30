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
from typing import Optional

import pandas as pd
import numpy
import gym
import torch
import torch.nn
import torch.nn.parallel
from torch.multiprocessing import Pool, set_start_method

from ppuu import configs
from ppuu import dataloader
from ppuu.lightning_modules.mpur import MPURModule

from ppuu.costs.policy_costs import PolicyCost


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


# class PolicyEvaluator:
#     def __init__(
#         self,
#         dataset: dataloader.EvaluationDataset,
#         env_map: str,
#         num_processes: int,
#     ):
#         self.dataset = dataset

#     def evaluate(model: torch.Module, output_dir: Optional[str]):
#         pass




@dataclass
class EvalConfig(configs.ConfigBase):
    I_80_PATH = "/misc/vlgscratch4/LecunGroup/nvidia-collab/traffic-data_offroad/state-action-cost/data_i80_v0/"
    checkpoint_path: str = None
    dataset: str = I_80_PATH
    env_map: str = "i80"
    save_sim_video: bool = False
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: str = None
    return_episode_data: bool = False

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


def process_one_episode(
    config,
    model_config,
    env,
    policy_model,
    policy_cost,
    data_stats,
    car_info,
    index,
):
    # movie_dir = path.join(
    #     opt.save_dir, "videos_simulator", plan_file, f"ep{index + 1}"
    # )
    # if opt.save_grad_vid:
    #     grad_movie_dir = path.join(
    #         opt.save_dir, "grad_videos_simulator", plan_file, f"ep{index + 1}"
    #     )
    #     print(f"[gradient videos will be saved to: {grad_movie_dir}]")

    inputs = env.reset(
        time_slot=car_info["time_slot"], vehicle_id=car_info["car_id"]
    )
    images, states, costs, actions = (
        [],
        [],
        [],
        [],
    )
    cntr = 0
    # inputs, cost, done, info = env.step(numpy.zeros((2,)))
    input_state_t0 = inputs["state"].contiguous()[-1]
    cost_sequence, action_sequence, state_sequence = [], [], []
    has_collided = False
    off_screen = False
    done = False

    # TODO: remove this
    policy_model.stats = data_stats
    policy_model.cuda()

    while not done:
        input_images = inputs["context"].contiguous()
        input_states = inputs["state"].contiguous()

        a = policy_model(
            input_images.cuda(),
            input_states.cuda(),
            sample=True,
            normalize_inputs=True,
            normalize_outputs=True,
        )
        a = a.cpu().view(1, 2).numpy()

        action_sequence.append(a)
        state_sequence.append(input_states)
        cntr += 1

        inputs, cost, done, info = env.step(a[0])
        if info.collisions_per_frame > 0:
            has_collided = True
            # print(f'[collision after {cntr} frames, ending]')
            done = True
        off_screen = info.off_screen
        images.append(input_images[-1])
        states.append(input_states[-1])
        costs.append([cost["pixel_proximity_cost"], cost["lane_cost"]])
        cost_sequence.append(cost)
        actions.append(
            ((torch.tensor(a[0]) - data_stats["a_mean"]) / data_stats["a_std"])
        )

    input_state_tfinal = inputs["state"][-1]

    images = torch.stack(images)
    states = torch.stack(states)
    costs = torch.tensor(costs)
    actions = torch.stack(actions)

    result = dict(
        time_travelled=len(images),
        distance_travelled=(input_state_tfinal[0] - input_state_t0[0]).item(),
        road_completed=1 if cost["arrived_to_dst"] else 0,
        off_screen=off_screen,
        has_collided=has_collided,
    )

    images_3_channels = (images[:, :3] + images[:, 3:]).clamp(max=255)
    episode_data = dict(
        result=result,
        action_sequence=numpy.stack(action_sequence),
        state_sequence=numpy.stack(state_sequence),
        cost_sequence=numpy.stack(cost_sequence),
        images=images_3_channels,
        gradients=None,
    )
    if config.save_gradients:
        episode_data["gradients"] = policy_cost.get_grad_vid(
            policy_model,
            dict(
                input_images=images[:, :3].contiguous(),
                input_states=states,
                car_sizes=torch.tensor(car_info["car_size"]),
            ),
        )[0]

    if config.output_dir is not None:
        episode_data_dir = os.path.join(config.output_dir, "episode_data")
        episode_output_path = os.path.join(episode_data_dir, str(index))
        torch.save(episode_data, episode_output_path)

    if config.return_episode_data:
        result.update(episode_data)

    return result


def get_performance_stats(results_per_episode):
    results_per_episode_df = pd.DataFrame.from_dict(
        results_per_episode, orient="index"
    )
    return dict(
        mean_distance=results_per_episode_df["distance_travelled"].mean(),
        mean_time=results_per_episode_df["time_travelled"].mean(),
        success_rate=results_per_episode_df["road_completed"].mean(),
    )


def main(config):
    torch.multiprocessing.set_sharing_strategy("file_system")

    mpur_module = MPURModule.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path
    )

    data_store = dataloader.DataStore(config.dataset)
    test_dataset = dataloader.EvaluationDataset(data_store, "test")

    i80_env_id = "I-80-v1"
    if i80_env_id not in [e.id for e in gym.envs.registry.all()]:
        gym.envs.registration.register(
            id=i80_env_id,
            entry_point="simulator.map_i80_ctrl:ControlledI80",
            kwargs=dict(
                fps=10,
                nb_states=mpur_module.config.model_config.n_cond,
                display=False,
                delta_t=0.1,
                store_simulator_video=config.save_sim_video,
                show_frame_count=False,
            ),
        )
    policy_cost = PolicyCost(
        mpur_module.config.cost_config, None, data_store.stats,
    )
    set_start_method("spawn")

    logging.info("Building the environment (loading data, if any)")
    env_names = {
        "i80": "I-80-v1",
    }
    env = gym.make(env_names["i80"])

    if config.num_processes > 0:
        executor = ProcessPoolExecutor(max_workers=config.num_processes)
    else:
        executor = ThreadPoolExecutor(max_workers=1)

    async_results = []
    time_started = time.time()

    os.makedirs(os.path.join(config.output_dir, "episode_data"), exist_ok=True)

    n_test = len(test_dataset)
    for j, data in enumerate(test_dataset):
        # async_results.append(
        #     pool.apply_async(
        #         process_one_episode,
        #         (
        #             config,
        #             mpur_module.config,
        #             env,
        #             mpur_module.policy_model,
        #             policy_cost,
        #             data_store.stats,
        #             data,
        #             j,
        #         ),
        #     )
        # )
        async_results.append(
            executor.submit(
                process_one_episode,
                config,
                mpur_module.config,
                env,
                mpur_module.policy_model,
                policy_cost,
                data_store.stats,
                data,
                j,
            )
        )

    results_per_episode = {}

    total_images = 0
    for j in range(n_test):
        simulation_result = async_results[j].result()
        results_per_episode[j] = simulation_result
        total_images += simulation_result["time_travelled"]
        stats = get_performance_stats(results_per_episode)

        log_string = " | ".join(
            (
                f"ep: {j + 1:3d}/{n_test}",
                f"time: {simulation_result['time_travelled']}",
                f"distance: {simulation_result['distance_travelled']:.0f}",
                f"success: {simulation_result['road_completed']:d}",
                f"success rate: {stats['success_rate']:.2f}",
            )
        )
        logging.info(log_string)

    executor.shutdown()

    diff_time = time.time() - time_started
    logging.info(
        f"avg time travelled per second is {total_images / diff_time}"
    )

    stats = get_performance_stats(results_per_episode)
    logging.info(f'mean distance travelled: {stats["mean_distance"]:.2f}')
    logging.info(f'mean time travelled: {stats["mean_time"]:.2f}')
    logging.info(f'success rate: {stats["success_rate"]:.2f}')

    result = dict(results_per_episode=results_per_episode, stats=stats,)
    if config.output_dir is not None:
        with open(
            os.path.join(config.output_dir, "evaluation_results.json"), "w"
        ) as f:
            json.dump(result, f, indent=4)
    return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalConfig.parse_from_command_line()
    main(config)
