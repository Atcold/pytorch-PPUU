import costs.policy_costs
import costs.policy_costs_continuous
import costs.policy_costs_km

PolicyCost = policy_costs.PolicyCost
PolicyCostContinuous = policy_costs_continuous.PolicyCostContinuous
PolicyCostKM = policy_costs_km.PolicyCostKM

MODEL_MAPPING = dict(
    vanilla=PolicyCost, continuous=PolicyCostContinuous, km=PolicyCostKM
)

def get_cost_model_from_name(name):
    return MODEL_MAPPING[name]
