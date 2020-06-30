"""Train a policy / controller"""

from dataclasses import dataclass

import torch

from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.lightning_modules.mpur import MPURModule, inject


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
        ).detach()
        z.requires_grad = True
        optimizer_z = self.get_z_optimizer(z)

        for i in range(self.config.training_config.n_z_updates):
            predictions = self.forward_model.unfold(
                self.policy_model, batch, z
            )
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "adv")
            optimizer_z.zero_grad()
            cost.backward()
            optimizer_z.step()

        return z

    def log_z(self, cost, components, t):
        if hasattr(self.logger, "log_custom"):
            self.logger.log_custom("z_cost", (cost.item(), t))
            self.logger.log_custom(
                "z_cost_proximity", (components["proximity_loss"].item(), t)
            )
            self.logger.log_custom(
                "z_cost_uncertainty", (components["u_loss"].item(), t)
            )

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
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "adv")
        else:
            predictions = self.forward(batch)
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "normal")
        loss = self.policy_cost.calculate_cost(batch, predictions)
        return {
            "loss": loss["policy_loss"],
            "log": loss,
            "progress_bar": loss,
        }
