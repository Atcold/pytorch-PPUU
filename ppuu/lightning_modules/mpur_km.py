"""Train a policy / controller"""


from ppuu.lightning_modules.mpur import MPURModule, inject
from ppuu.costs.policy_costs_km import PolicyCostKM, PolicyCostKMSplit
from ppuu.modeling.forward_model_km import ForwardModelKM


@inject(cost_type=PolicyCostKM, fm_type=ForwardModelKM)
class MPURKMModule(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold_km(self.policy_model, batch)
        return predictions


@inject(cost_type=PolicyCostKMSplit, fm_type=ForwardModelKM)
class MPURKMSplitModule(MPURKMModule):
    pass
