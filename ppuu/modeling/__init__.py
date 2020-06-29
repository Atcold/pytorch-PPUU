from ppuu.modeling.forward_models import ForwardModel
from ppuu.modeling.forward_model_km import ForwardModelKM

FM_MAPPING = dict(vanilla=ForwardModel, km=ForwardModelKM,)


def get_forward_model_from_name(name):
    return FM_MAPPING[name]
