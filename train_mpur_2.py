"""Train a policy / controller"""
import dataclasses
from dataclasses import dataclass, field
import os
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataloader import DataStore, Dataset
import policy_costs
import configs
import policy_models

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MPURModule(pl.LightningModule):
    @dataclass
    class Config(configs.ConfigBase):
        forward_model_path: str = "/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/offroad/model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model"

        n_cond: int = 20
        n_pred: int = 30
        n_hidden: int = 256
        n_feature: int = 256

        n_inputs: int = 4
        n_actions: int = 2
        height: int = 117
        width: int = 24
        h_height: int = 14
        h_width: int = 3
        hidden_size: int = n_feature * h_height * h_width

    def __init__(self, hparams=None):
        super().__init__()
        if type(hparams) is dict:
            self.hparams = hparams
            self.config = MasterConfig.parse_from_dict(hparams)
        else:
            self.hparams = dataclasses.asdict(hparams)
            self.config = hparams

        model = torch.load(self.config.model_config.forward_model_path)
        if type(model) is dict:
            model = model["model"]
        if not hasattr(model.encoder, "n_channels"):
            model.encoder.n_channels = 3

        self.forward_model = model
        self.policy_model = policy_models.DeterministicPolicy(
            n_cond=self.config.model_config.n_cond,
            n_feature=self.config.model_config.n_feature,
            n_actions=self.config.model_config.n_actions,
            h_height=self.config.model_config.h_height,
            h_width=self.config.model_config.h_width,
            n_hidden=self.config.model_config.n_hidden,
        )
        self.policy_model.train()

    def forward(self, batch):
        self.forward_model.eval()
        predictions = policy_costs.policy_unfold(
            self.forward_model, self.policy_model, batch
        )
        total_u_loss = policy_costs.compute_uncertainty_batch(
            self.forward_model,
            dict(
                input_images=batch["input_images"],
                input_states=batch["input_states"],
                actions=predictions["pred_actions"],
                car_sizes=batch["car_sizes"],
                ego_cars=batch["ego_cars"],
            ),
            cost_config=self.config.cost_config,
            data_stats=self.data_store.stats,
            npred=self.config.model_config.n_pred,
            n_models=self.config.cost_config.n_models,
            Z=predictions["Z"],
            estimation=False,
        )["total_u_loss"]

        loss_a = predictions["pred_actions"].norm(2, 2).pow(2).mean()
        state_losses = policy_costs.compute_state_costs(
            predictions["pred_images"],
            predictions["pred_states"],
            batch["car_sizes"],
            self.data_store.stats,
        )

        result = dict(
            proximity_loss=state_losses["proximity_loss"],
            lane_loss=state_losses["lane_loss"],
            offroad_loss=state_losses["offroad_loss"],
            uncertainty_loss=total_u_loss,
            action_loss=loss_a,
        )
        result["policy_loss"] = policy_costs.compute_combined_loss(
            self.config.cost_config, **result
        )
        return result

    def training_step(self, batch, batch_idx):
        result = self(batch)
        return {
            "loss": result["policy_loss"],
            "log": result,
            "progress_bar": result,
        }

    def validation_step(self, batch, batch_idx):
        if hasattr(self.forward_model, "u_costs_mean"):
            result = self(batch)
            return {
                "val_loss": result["policy_loss"],
                "log": result,
                "progress_bar": result,
            }
        else:
            return {"val_loss": torch.Tensor(0)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.policy_model.parameters(),
            self.config.training_config.learning_rate,
        )
        return optimizer

    def prepare_data(self):
        self.data_store = DataStore("i80")
        # self.data_store = DataStore(
        #     "/home/us441/nvidia-collab/vlad/traffic-data_offroad_small_sparse_dense/state-action-cost/data_i80_v0/"
        # )
        samples_in_epoch = (
            self.config.training_config.epoch_size
            * self.config.training_config.batch_size
        )
        samples_in_validation = (
            self.config.training_config.validation_size
            * self.config.training_config.batch_size
        )
        self.train_dataset = Dataset(
            self.data_store, "train", 20, 30, size=samples_in_epoch
        )
        self.val_dataset = Dataset(
            self.data_store, "val", 20, 30, size=samples_in_validation
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training_config.batch_size,
            num_workers=8,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training_config.batch_size,
            num_workers=8,
        )
        return loader

    def on_train_start(self):
        self.forward_model.eval()
        self.forward_model.device = self.device
        self.forward_model.to(self.device)
        policy_costs.estimate_uncertainty_stats(
            self.forward_model,
            self.train_dataloader(),
            self.config.cost_config,
            self.data_store.stats,
            n_batches=10,
            npred=self.config.model_config.n_pred,
        )


@dataclass
class CostConfig(configs.ConfigBase):
    """Configuration of cost calculation"""

    u_reg: float = field(default=0.05)
    lambda_a: float = field(default=0.0)
    lambda_l: float = field(default=0.2)
    lambda_o: float = field(default=1.0)
    lambda_p: float = field(default=1.0)
    gamma: float = field(default=0.99)
    u_hinge: float = field(default=0.5)
    n_models: int = field(default=10)


@dataclass
class MasterConfig(configs.ConfigBase):
    model_config: MPURModule.Config
    cost_config: CostConfig
    training_config: configs.TrainingConfig


class JsonLogger(pl.loggers.TensorBoardLogger):
    def __init__(self, *args, json_filename="logs.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
        self.json_filename = json_filename

    @pl.loggers.base.rank_zero_only
    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)
        self.logs.append(metrics)

    @pl.loggers.base.rank_zero_only
    def save(self):
        super().save()
        logs_json_save_path = os.path.join(self.log_dir, self.json_filename)
        with open(logs_json_save_path, "w") as f:
            json.dump(self.logs, f, indent=4)


def main(config):
    pl.seed_everything(config.training_config.seed)

    logger = JsonLogger(
        save_dir=config.training_config.output_dir,
        name=config.training_config.experiment_name,
        version=f"seed={config.training_config.seed}",
    )

    trainer = pl.Trainer(
        gpus=1,
        grad_clip_val=50.0,
        max_epochs=config.training_config.n_epochs,
        check_val_every_n_epoch=1,
        add_log_row_interval=10,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(save_top_k=-1, period=20),
        logger=logger,
    )
    model = MPURModule(config)
    trainer.fit(model)


if __name__ == "__main__":
    config = MasterConfig.parse_from_command_line()
    print("parsed config", config)
    main(config)
