import random
import os
import re
import logging

import torch


class DataStore:
    def __init__(self, dataset):
        if dataset == "i80" or dataset == "us101":
            data_dir = f"traffic-data/state-action-cost/data_{dataset}_v0"
        else:
            data_dir = dataset

        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
        self.ids = []
        self.ego_car_images = []
        data_files = next(os.walk(data_dir))[1]
        for df in data_files:
            combined_data_path = f"{data_dir}/{df}/all_data.pth"
            logging.info(f"Loading pickle {combined_data_path}")
            if os.path.isfile(combined_data_path):
                data = torch.load(combined_data_path)
                self.images += data.get("images")
                self.actions += data.get("actions")
                self.costs += data.get("costs")
                self.states += data.get("states")
                self.ids += data.get("ids")
                self.ego_car_images += data.get("ego_car")

        self.n_episodes = len(self.images)
        splits_path = data_dir + "/splits.pth"
        if os.path.exists(splits_path):
            logging.info(f"Loading splits {splits_path}")
            splits = torch.load(splits_path)
            self.splits = dict(
                train=splits.get("train_indx"),
                val=splits.get("valid_indx"),
                test=splits.get("test_indx"),
            )

        stats_path = data_dir + "/data_stats.pth"
        if os.path.isfile(stats_path):
            logging.info(f"Loading data stata {stats_path}")
            stats = torch.load(stats_path)
            self.stats = stats
            self.a_mean = stats.get("a_mean")
            self.a_std = stats.get("a_std")
            self.s_mean = stats.get("s_mean")
            self.s_std = stats.get("s_std")

        car_sizes_path = data_dir + "/car_sizes.pth"
        self.car_sizes = torch.load(car_sizes_path)

    def parse_car_path(path):
        splits = path.split("/")
        time_slot = splits[-2]
        car_id = int(re.findall("car(\d+).pkl", splits[-1])[0])
        data_files = {
            "trajectories-0400-0415": 0,
            "trajectories-0500-0515": 1,
            "trajectories-0515-0530": 2,
        }
        time_slot = data_files[time_slot]
        return time_slot, car_id

    def get_episode_car_info(self, episode):
        splits = self.ids[episode].split("/")
        time_slot_str = splits[-2]
        car_id = int(re.findall("car(\d+).pkl", splits[-1])[0])
        data_files_mapping = {
            "trajectories-0400-0415": 0,
            "trajectories-0500-0515": 1,
            "trajectories-0515-0530": 2,
        }
        time_slot = data_files_mapping[time_slot_str]
        car_size = self.car_sizes[time_slot_str][car_id]
        result = dict(
            time_slot=time_slot,
            time_slot_str=time_slot_str,
            car_id=car_id,
            car_size=car_size,
        )
        return result


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_store, split, n_cond, n_pred, size):
        self.split = split
        self.data_store = data_store
        self.n_cond = n_cond
        self.n_pred = n_pred
        self.size = size
        self.random = random.Random()
        self.random.seed(12345)

    def sample_episode(self):
        return self.random.choice(self.data_store.splits[self.split])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.get_one_example()

    def get_one_example(self):
        T = self.n_cond + self.n_pred
        while True:
            s = self.sample_episode()
            # s = indx[0]
            # min is important since sometimes numbers do not align causing
            # issues in stack operation below
            episode_length = min(
                self.data_store.images[s].size(0),
                self.data_store.states[s].size(0),
            )
            if episode_length >= T:
                t = self.random.randint(0, episode_length - T)
                images = self.data_store.images[s][t : t + T]
                actions = self.data_store.actions[s][t : t + T]
                states = self.data_store.states[s][t : t + T, 0]
                costs = self.data_store.costs[s][t : t + T]
                ids = self.data_store.ids[s]
                ego_cars = self.data_store.ego_car_images[s]
                splits = self.data_store.ids[s].split("/")
                time_slot = splits[-2]
                car_id = int(re.findall(r"car(\d+).pkl", splits[-1])[0])
                size = self.data_store.car_sizes[time_slot][car_id]
                car_sizes = torch.tensor([size[0], size[1]])
                break

        actions = self.normalise_action(actions.clone())
        states = self.normalise_state_vector(states.clone())
        images = self.normalise_state_image(images.clone())
        ego_cars = self.normalise_state_image(ego_cars.clone())

        t0 = self.n_cond
        t1 = T
        input_images = images[:t0].float().contiguous()
        input_states = states[:t0].float().contiguous()
        target_images = images[t0:t1].float().contiguous()
        target_states = states[t0:t1].float().contiguous()
        target_costs = costs[t0:t1].float().contiguous()
        t0 -= 1
        t1 -= 1
        actions = actions[t0:t1].float().contiguous()
        ego_cars = ego_cars.float().contiguous()
        #          n_cond                      n_pred
        # <---------------------><---------------------------------->
        # .                     ..                                  .
        # +---------------------+.                                  .  ^          ^
        # |i|i|i|i|i|i|i|i|i|i|i|.  3 × 117 × 24                    .  |          |
        # +---------------------+.                                  .  | inputs   |
        # +---------------------+.                                  .  |          |
        # |s|s|s|s|s|s|s|s|s|s|s|.  4                               .  |          |
        # +---------------------+.                                  .  v          |
        # .                   +-----------------------------------+ .  ^          |
        # .                2  |a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| .  | actions  |
        # .                   +-----------------------------------+ .  v          |
        # .                     +-----------------------------------+  ^          | tensors
        # .       3 × 117 × 24  |i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|  |          |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  4  |s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|  | targets  |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  2  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|  |          |
        # .                     +-----------------------------------+  v          v
        # +---------------------------------------------------------+             ^
        # |                           car_id                        |             | string
        # +---------------------------------------------------------+             v
        # +---------------------------------------------------------+             ^
        # |                          car_size                       |  2          | tensor
        # +---------------------------------------------------------+             v

        return dict(
            input_images=input_images,
            input_states=input_states,
            ego_cars=ego_cars,
            actions=actions,
            target_images=target_images,
            target_states=target_states,
            target_costs=target_costs,
            ids=ids,
            car_sizes=car_sizes,
        )

    @staticmethod
    def normalise_state_image(images):
        return images.float().div_(255.0)

    def normalise_state_vector(self, states):
        shape = (
            (1, 1, 4) if states.dim() == 3 else (1, 4)
        )  # dim = 3: state sequence, dim = 2: single state
        states -= (
            self.data_store.s_mean.view(*shape)
            .expand(states.size())
            .to(states.device)
        )
        states /= (
            1e-8 + self.data_store.s_std.view(*shape).expand(states.size())
        ).to(states.device)
        return states

    def normalise_action(self, actions):
        actions -= (
            self.data_store.a_mean.view(1, 2)
            .expand(actions.size())
            .to(actions.device)
        )
        actions /= (
            1e-8 + self.data_store.a_std.view(1, 2).expand(actions.size())
        ).to(actions.device)
        return actions


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, data_store, split="test"):
        self.split = split
        self.data_store = data_store
        self.size = len(self.data_store.splits[self.split])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        car_info = self.data_store.get_episode_car_info(
            self.data_store.splits[self.split][i]
        )
        return car_info


if __name__ == "__main__":
    ds = DataStore("i80")
    d = Dataset(ds, "train", 20, 30, 100)
