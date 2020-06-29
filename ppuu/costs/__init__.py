from ppuu.costs.policy_costs import PolicyCost
from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.costs.policy_costs_km import PolicyCostKM

MODEL_MAPPING = dict(
    vanilla=PolicyCost, continuous=PolicyCostContinuous, km=PolicyCostKM
)

def get_cost_model_from_name(name):
    return MODEL_MAPPING[name]
