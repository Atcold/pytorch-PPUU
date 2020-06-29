import argparse

from ppuu.lightning_modules.mpur import MPURModule, MPURContinuousModule
from ppuu.lightning_modules.mpur_dreaming import MPURDreamingModule
from ppuu.lightning_modules.mpur_km import MPURKMModule, MPURKMSplitModule

MODULES_DICT = dict(
    vanilla=MPURModule,
    dreaming=MPURDreamingModule,
    km=MPURKMModule,
    km_split=MPURKMSplitModule,
    continuous=MPURContinuousModule,
)


def get_module_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        help="Pick the model type to run",
        required=True,
    )
    args, _ = parser.parse_known_args()
    return get_module(args.model_type)


def get_module(name):
    return MODULES_DICT[name]
