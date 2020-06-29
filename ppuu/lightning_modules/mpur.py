"""Train a policy / controller"""
import dataclasses
from dataclasses import dataclass
import hashlib


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ppuu.dataloader import DataStore, Dataset
from ppuu.costs import PolicyCost, PolicyCostContinuous
from ppuu import configs
from ppuu.modeling import policy_models
from ppuu.modeling.forward_models import ForwardModel


def inject(cost_type=PolicyCost, fm_type=ForwardModel):
    """ This injector allows to customize lightning modules with custom cost
    class and forward model class (or more, if extended).  It injects these
    types and creates a config dataclass that contains configs for all the
    components of this lightning module. Any new module has to be injected
    into, as without it the class doesn't know which cost or forward model to
    use, and also has no config.

    The config class has to be added to the global scope through globals().
    It's a hack to make it pickleable for multiprocessing later.
    If the MPURModule has to be pickleable too, we'd need to put it into
    global scope too.
    """

    def wrapper(cls_):
        h = hashlib.md5(
            (cost_type.__qualname__ + fm_type.__qualname__).encode()
        ).hexdigest()[:7]
        suffix = f"{cls_.__name__}_{cost_type.__name__}_{fm_type.__name__}_{h}"
        config_name = f"config_{suffix}"

        class Cls(cls_):
            CostType = cost_type
            ForwardModelType = fm_type

            @dataclass
            class Config(configs.ConfigBase):
                model_config: cls_.ModelConfig = cls_.ModelConfig()
                cost_config: cost_type.Config = cost_type.Config()
                training_config: cls_.TrainingConfig = cls_.TrainingConfig()

        Cls.Config.__qualname__ = config_name
        Cls.Config.__name__ = config_name
        globals()[config_name] = Cls.Config
        return Cls

    return wrapper


@inject(cost_type=PolicyCost, fm_type=ForwardModel)
class MPURModule(pl.LightningModule):
    @dataclass
    class ModelConfig(configs.ModelConfig):
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

    TrainingConfig = configs.TrainingConfig

    def __init__(self, hparams=None):
        super().__init__()
        self.set_hparams(hparams)

        self.forward_model = self.ForwardModelType(
            self.config.model_config.forward_model_path
        )
        self.policy_model = policy_models.DeterministicPolicy(
            n_cond=self.config.model_config.n_cond,
            n_feature=self.config.model_config.n_feature,
            n_actions=self.config.model_config.n_actions,
            h_height=self.config.model_config.h_height,
            h_width=self.config.model_config.h_width,
            n_hidden=self.config.model_config.n_hidden,
        )
        self.policy_model.train()

    def set_hparams(self, hparams=None):
        if hparams is None:
            hparams = MPURModule.Config()
        if type(hparams) is dict:
            self.hparams = hparams
            self.config = MPURModule.Config.parse_from_dict(hparams)
        else:
            self.hparams = dataclasses.asdict(hparams)
            self.config = hparams

    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold(self.policy_model, batch)
        return predictions

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        return {
            "loss": loss["policy_loss"],
            "log": loss,
            "progress_bar": loss,
        }

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        return {
            "val_loss": loss["policy_loss"],
            "log": loss,
            "progress_bar": loss,
        }

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
        self.data_store = DataStore(self.config.training_config.dataset)

        def worker_init_fn(index):
            info = torch.utils.data.get_worker_info()
            info.dataset.random.seed(info.seed)

        self.worker_init_fn = worker_init_fn

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
            worker_init_fn=self.worker_init_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training_config.batch_size,
            num_workers=8,
            worker_init_fn=self.worker_init_fn,
        )
        return loader

    def on_train_start(self):
        self.forward_model.eval()
        self.forward_model.device = self.device
        self.forward_model.to(self.device)

        self.policy_cost = self.CostType(
            self.config.cost_config, self.forward_model, self.data_store.stats
        )
        self.policy_cost.estimate_uncertainty_stats(self.train_dataloader())

    @classmethod
    def _load_model_state(cls, checkpoint, *args, **kwargs):
        copy = dict()
        for k in checkpoint["state_dict"]:
            if k.startswith("forward_model") and not k.startswith(
                "forward_model.forward_model."
            ):
                copy["forward_model." + k] = checkpoint["state_dict"][k]
            else:
                copy[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = copy
        return super()._load_model_state(checkpoint, *args, **kwargs)


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModel)
class MPURContinuousModule(MPURModule):
    pass
