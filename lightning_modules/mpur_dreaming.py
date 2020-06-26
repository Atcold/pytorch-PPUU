"""Train a policy / controller"""

from dataclasses import dataclass
import logging

import torch

import configs
from costs.policy_costs_continuous import PolicyCostContinuous
from lightning_modules.mpur import MPURModule, inject


@inject(cost_type=PolicyCostContinuous)
class MPURDreamingModule(MPURModule):
    @dataclass
    class TrainingConfig(MPURModule.TrainingConfig):
        lrt_z: float = 0.1
        n_z_updates: int = 10
        adversarial_frequency: int = 10

    def get_adversarial_z(self, batch):
        z = self.forward_model.sample_z(
            self.config.model_config.n_pred
            * self.config.training_config.batch_size
        )
        z = z.view(
            self.config.training_config.batch_size,
            self.config.model_config.n_pred,
            -1,
        )
        optimizer_z = self.get_z_optimizer(z)

        for i in range(self.config.training_config.n_z_updates):
            predictions = self.forward_model.unfold(
                self.policy_model, batch, z
            )
            cost = self.policy_cost.calculate_z_cost(batch, predictions)
            self.logger.log_custom("z_cost", (cost.item(), "adv"))
            optimizer_z.zero_grad()
            cost.backward()
            optimizer_z.step()

        return z

    def get_z_optimizer(self, Z):
        return torch.optim.Adam([Z], self.config.training_config.lrt_z)

    def forward_adversarial(self, batch):
        self.forward_model.eval()
        z = self.get_adversarial_z(batch)
        predictions = self.forward_model.unfold(self.policy_model, batch, z)
        return predictions

    def training_step(self, batch, batch_idx):
        if batch_idx % self.config.training_config.adversarial_frequency == 0:
            predictions = self.forward_adversarial(batch)
            self.logger.log_custom(
                "z_cost",
                (
                    self.policy_cost.calculate_z_cost(
                        batch, predictions
                    ).item(),
                    "adv",
                ),
            )
        else:
            predictions = self.forward(batch)
            self.logger.log_custom(
                "z_cost",
                (
                    self.policy_cost.calculate_z_cost(
                        batch, predictions
                    ).item(),
                    "normal",
                ),
            )
        loss = self.policy_cost.calculate_cost(batch, predictions)
        return {
            "loss": loss["policy_loss"],
            "log": loss,
            "progress_bar": loss,
        }
